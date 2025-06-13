"use client"

import { useState } from 'react';
import { Search, Zap } from 'lucide-react';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { useDashboard } from './dashboard-context';
import { apiService } from '@/lib/api-service';

export function RepositoryInputForm() {
  const { 
    repoUrl, 
    setRepoUrl, 
    setRepoData, 
    setIsLoading, 
    setError, 
    analysisMode, 
    setAnalysisMode,
    setCommitData,
    setActiveView
  } = useDashboard();

  const handleFetchRepo = async () => {
    if (!repoUrl.trim()) {
      setError("Please enter a repository URL or owner/repo");
      return;
    }

    setIsLoading(true);
    setError('');
    
    try {
      const data = await apiService.analyzeRepo(repoUrl);
      
      setRepoData({
        name: repoUrl,
        description: data.description,
        linesOfCode: data.line_metrics.total.loc,
        cyclomaticComplexity: data.cyclomatic_complexity.average,
        depthOfInheritance: data.depth_of_inheritance.average,
        halsteadVolume: data.halstead_metrics.total_volume,
        maintainabilityIndex: data.maintainability_index.average,
        commentDensity: data.line_metrics.total.comment_density,
        sloc: data.line_metrics.total.sloc,
        lloc: data.line_metrics.total.lloc,
        numberOfFiles: data.num_files,
        numberOfFunctions: data.num_functions,
        numberOfClasses: data.num_classes,
      });

      const transformedCommitData = Object.entries(data.monthly_commits)
        .map(([date, commits]) => ({
          month: new Date(date).toLocaleString('default', { month: 'long' }),
          commits: commits as number,
        }))
        .slice(0, 12)
        .reverse();

      setCommitData(transformedCommitData);
      setActiveView('metrics');
    } catch (error) {
      console.error('Error fetching repo data:', error);
      setError('Error fetching repository data. Please check the URL and try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      handleFetchRepo();
    }
  };

  return (
    <div className="flex gap-2 items-center">
      <Input
        type="text"
        placeholder="Enter the GitHub repo link or owner/repo"
        value={repoUrl}
        onChange={(e) => setRepoUrl(e.target.value)}
        onKeyPress={handleKeyPress}
        className="flex-1"
        title="Format: https://github.com/owner/repo or owner/repo"
      />
      <Select value={analysisMode} onValueChange={setAnalysisMode}>
        <SelectTrigger className="w-48">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="error_focused">Error Focused</SelectItem>
          <SelectItem value="dependency_focused">Dependency Focused</SelectItem>
        </SelectContent>
      </Select>
      <Button 
        onClick={handleFetchRepo} 
        className="flex items-center gap-2"
      >
        <Zap className="h-4 w-4" />
        Analyze
      </Button>
    </div>
  );
}

