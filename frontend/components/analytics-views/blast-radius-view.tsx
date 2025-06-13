"use client"

import { useState } from 'react';
import { useDashboard } from '@/components/dashboard-context';
import { apiService } from '@/lib/api-service';
import { Target, AlertTriangle, Bug, Code2, FileCode } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Progress } from '@/components/ui/progress';

export function BlastRadiusView() {
  const { 
    repoUrl, 
    blastRadiusData, 
    setBlastRadiusData, 
    isLoading, 
    setIsLoading, 
    error, 
    setError,
    selectedSymbol,
    setSelectedSymbol
  } = useDashboard();

  const handleBlastRadiusAnalysis = async () => {
    if (!selectedSymbol.trim()) {
      setError("Please enter a symbol name for blast radius analysis");
      return;
    }

    setIsLoading(true);
    setError("");
    
    try {
      const data = await apiService.analyzeBlastRadius(repoUrl, selectedSymbol);
      setBlastRadiusData(data);
    } catch (error) {
      console.error('Error during blast radius analysis:', error);
      setError('Failed to analyze symbol impact. Please check the symbol name and try again.');
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
              <span>Analyzing symbol impact...</span>
              <span>Please wait</span>
            </div>
            <Progress value={undefined} className="w-full" />
          </div>
        </CardContent>
      </Card>
    );
  }

  if (!blastRadiusData) {
    return (
      <div className="space-y-4">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Target className="h-5 w-5" />
              Blast Radius Analysis
            </CardTitle>
            <CardDescription>
              Analyze the impact of changes to a specific symbol
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex gap-2">
              <Input
                placeholder="Symbol name for blast radius analysis"
                value={selectedSymbol}
                onChange={(e) => setSelectedSymbol(e.target.value)}
                className="flex-1"
              />
              <Button 
                onClick={handleBlastRadiusAnalysis} 
                className="flex items-center gap-2"
              >
                <Target className="h-4 w-4" />
                Analyze Impact
              </Button>
            </div>
            
            {error && (
              <Alert variant="destructive">
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
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Target className="h-5 w-5 text-purple-500" />
            Symbol Impact Analysis
          </CardTitle>
          <CardDescription>
            Understanding the blast radius of changes to {blastRadiusData.target_symbol.name}
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <Card>
              <CardContent className="pt-6">
                <div className="text-2xl font-bold">{blastRadiusData.blast_radius.affected_nodes}</div>
                <p className="text-xs text-muted-foreground">Affected Nodes</p>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="pt-6">
                <div className="text-2xl font-bold">{blastRadiusData.blast_radius.affected_edges}</div>
                <p className="text-xs text-muted-foreground">Affected Edges</p>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="pt-6">
                <div className="text-2xl font-bold">{blastRadiusData.target_symbol.issues.length}</div>
                <p className="text-xs text-muted-foreground">Issues Found</p>
              </CardContent>
            </Card>
          </div>

          <div className="border rounded-lg p-4">
            <h4 className="font-medium mb-2">Target Symbol Details</h4>
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <Badge variant="outline">{blastRadiusData.target_symbol.type}</Badge>
                <span className="font-medium">{blastRadiusData.target_symbol.name}</span>
              </div>
              <p className="text-sm text-muted-foreground">{blastRadiusData.target_symbol.path}</p>
              
              {blastRadiusData.target_symbol.issues.length > 0 && (
                <div className="space-y-2 mt-3">
                  <h5 className="text-sm font-medium">Issues:</h5>
                  {blastRadiusData.target_symbol.issues.map((issue, index) => (
                    <div key={index} className="flex items-start gap-2 text-sm">
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
                </div>
              )}
            </div>
          </div>

          <div className="flex gap-2">
            <Input
              placeholder="Symbol name for blast radius analysis"
              value={selectedSymbol}
              onChange={(e) => setSelectedSymbol(e.target.value)}
              className="flex-1"
            />
            <Button 
              onClick={handleBlastRadiusAnalysis} 
              className="flex items-center gap-2"
            >
              <Target className="h-4 w-4" />
              Analyze Impact
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

