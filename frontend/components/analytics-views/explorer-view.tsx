"use client"

import { useState, useEffect } from 'react';
import { useDashboard } from '@/components/dashboard-context';
import { apiService } from '@/lib/api-service';
import { Search, Code2, AlertTriangle, Target, Zap, FileCode, Bug } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Progress } from '@/components/ui/progress';
import { Issue } from '@/lib/api-types';

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

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical': return 'bg-red-500';
      case 'major': return 'bg-orange-500';
      case 'minor': return 'bg-yellow-500';
      default: return 'bg-gray-500';
    }
  };

  const getIssueTypeIcon = (type: string) => {
    switch (type) {
      case 'unused_parameter':
      case 'unused_function':
        return <Code2 className="h-4 w-4" />;
      case 'mutable_default_parameter':
      case 'missing_required_arguments':
        return <AlertTriangle className="h-4 w-4" />;
      case 'undefined_function_call':
        return <Bug className="h-4 w-4" />;
      default:
        return <FileCode className="h-4 w-4" />;
    }
  };

  if (isLoading) {
    return (
      <Card>
        <CardContent className="pt-6">
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span>Analyzing codebase...</span>
              <span>Please wait</span>
            </div>
            <Progress value={undefined} className="w-full" />
          </div>
        </CardContent>
      </Card>
    );
  }

  if (!explorationData) {
    return (
      <div className="space-y-4">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Search className="h-5 w-5" />
              Code Explorer
            </CardTitle>
            <CardDescription>
              Analyze codebases for functional errors, parameter issues, and architectural problems
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Button 
              onClick={handleVisualExploration} 
              className="flex items-center gap-2"
            >
              <Zap className="h-4 w-4" />
              Start Visual Analysis
            </Button>
            
            {error && (
              <Alert variant="destructive" className="mt-4">
                <AlertTriangle className="h-4 w-4" />
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="pt-6">
            <div className="text-2xl font-bold">{explorationData.summary.total_nodes}</div>
            <p className="text-xs text-muted-foreground">Total Nodes</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <div className="text-2xl font-bold text-red-600">{explorationData.summary.total_issues}</div>
            <p className="text-xs text-muted-foreground">Functional Issues</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <div className="text-2xl font-bold text-orange-600">{explorationData.summary.error_hotspots_count}</div>
            <p className="text-xs text-muted-foreground">Error Hotspots</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <div className="text-2xl font-bold text-blue-600">{explorationData.summary.critical_paths_count}</div>
            <p className="text-xs text-muted-foreground">Critical Paths</p>
          </CardContent>
        </Card>
      </div>

      {/* Error Hotspots */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <AlertTriangle className="h-5 w-5 text-red-500" />
            Error Hotspots
          </CardTitle>
          <CardDescription>
            Functions and classes with the most functional issues
          </CardDescription>
        </CardHeader>
        <CardContent>
          <ScrollArea className="h-96">
            <div className="space-y-4">
              {explorationData.error_hotspots.slice(0, 10).map((hotspot, index) => (
                <div key={hotspot.id} className="border rounded-lg p-4 space-y-2">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <Badge variant="outline">{hotspot.type}</Badge>
                      <span className="font-medium">{hotspot.name}</span>
                    </div>
                    <Badge variant="destructive">{hotspot.issues.length} issues</Badge>
                  </div>
                  <p className="text-sm text-muted-foreground">{hotspot.path}</p>
                  
                  {hotspot.issues.length > 0 && (
                    <div className="space-y-1">
                      {hotspot.issues.slice(0, 3).map((issue, issueIndex) => (
                        <div key={issueIndex} className="flex items-start gap-2 text-sm">
                          {getIssueTypeIcon(issue.type)}
                          <div className="flex-1">
                            <div className="flex items-center gap-2">
                              <Badge 
                                variant="outline" 
                                className={`text-white ${getSeverityColor(issue.severity)}`}
                              >
                                {issue.severity}
                              </Badge>
                              <span className="font-medium">{issue.type}</span>
                            </div>
                            <p className="text-muted-foreground mt-1">{issue.message}</p>
                            <p className="text-blue-600 text-xs mt-1">💡 {issue.suggestion}</p>
                          </div>
                        </div>
                      ))}
                      {hotspot.issues.length > 3 && (
                        <p className="text-xs text-muted-foreground">
                          +{hotspot.issues.length - 3} more issues...
                        </p>
                      )}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </ScrollArea>
        </CardContent>
      </Card>

      {/* Insights */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Zap className="h-5 w-5 text-blue-500" />
            Analysis Insights
          </CardTitle>
          <CardDescription>
            Key findings and recommendations
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {explorationData.exploration_insights.map((insight, index) => (
              <div key={index} className="border rounded-lg p-4">
                <div className="flex items-center gap-2 mb-2">
                  <Badge 
                    variant={insight.priority === 'critical' ? 'destructive' : 
                           insight.priority === 'major' ? 'default' : 'secondary'}
                  >
                    {insight.priority}
                  </Badge>
                  <span className="font-medium">{insight.title}</span>
                </div>
                <p className="text-sm text-muted-foreground">{insight.description}</p>
                {insight.affected_nodes.length > 0 && (
                  <p className="text-xs text-blue-600 mt-2">
                    Affects {insight.affected_nodes.length} nodes
                  </p>
                )}
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

