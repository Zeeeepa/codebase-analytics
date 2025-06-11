"use client"

import { useState, useEffect } from "react"
import { BarChart3, Code2, FileCode2, GitBranch, Github, Settings, MessageSquare, FileText, Code, RefreshCcw, PaintBucket, Brain, AlertTriangle, Shield, Zap, TrendingUp, Network } from "lucide-react"
import { Bar, BarChart, ResponsiveContainer, XAxis, YAxis, PieChart, Pie, Cell, LineChart, Line, Tooltip, Legend, ScatterChart, Scatter } from "recharts"

import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Progress } from "@/components/ui/progress"
import UpgradeRunUI, { UpgradeRunResults } from "@/components/upgrade-run-ui"

// Enhanced interfaces for the new API response
interface CodeIssue {
  type: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  message: string;
  file_path: string;
  line_number?: number;
  function_name?: string;
  suggestion: string;
  impact: string;
  category: string;
}

interface FileMetric {
  path: string;
  name: string;
  loc: number;
  complexity: number;
  maintainability: number;
  risk_score: number;
  functions: number;
  classes: number;
}

interface DependencyNode {
  id: string;
  label: string;
  type: 'file' | 'function';
  size: number;
  complexity: number;
  group: string;
}

interface EnhancedRepoData {
  repo_url: string;
  description: string;
  quality_score: number;
  quality_grade: string;
  line_metrics: {
    total: {
      loc: number;
      sloc: number;
      lloc: number;
      comments: number;
      comment_density: number;
    }
  };
  cyclomatic_complexity: { average: number; total: number };
  maintainability_index: { average: number };
  halstead_metrics: { total_volume: number; average_volume: number };
  depth_of_inheritance: { average: number };
  num_files: number;
  num_functions: number;
  num_classes: number;
  monthly_commits: Record<string, number>;
  issues: {
    statistics: {
      total: number;
      critical: number;
      high: number;
      medium: number;
      low: number;
      by_category: Record<string, number>;
    };
    details: CodeIssue[];
  };
  visualizations: {
    dependency_graph: {
      nodes: DependencyNode[];
      edges: Array<{ from: string; to: string; type: string }>;
      stats: Record<string, number>;
    };
    complexity_heatmap: {
      files: Array<{
        file: string;
        total_complexity: number;
        avg_complexity: number;
        lines: number;
        risk_level: 'low' | 'medium' | 'high';
      }>;
      summary: Record<string, number>;
    };
    file_metrics: FileMetric[];
    risk_distribution: {
      high_risk_files: number;
      medium_risk_files: number;
      low_risk_files: number;
    };
  };
}

const SEVERITY_COLORS = {
  critical: '#dc2626',
  high: '#ea580c',
  medium: '#d97706',
  low: '#65a30d'
};

const CATEGORY_COLORS = {
  security: '#dc2626',
  complexity: '#7c3aed',
  maintainability: '#2563eb',
  style: '#059669',
  performance: '#db2777'
};

export default function EnhancedAnalyticsDashboard() {
  const [repoUrl, setRepoUrl] = useState("")
  const [repoData, setRepoData] = useState<EnhancedRepoData | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [isLandingPage, setIsLandingPage] = useState(true)
  const [selectedIssue, setSelectedIssue] = useState<CodeIssue | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [isMounted, setIsMounted] = useState(false)

  // Fix hydration mismatch by ensuring component is only rendered client-side
  useEffect(() => {
    setIsMounted(true)
  }, [])

  // Return null during server-side rendering or before hydration
  if (!isMounted) {
    return null;
  }

  const parseRepoUrl = (input: string): string => {
    if (input.includes('github.com')) {
      const url = new URL(input);
      const pathParts = url.pathname.split('/').filter(Boolean);
      if (pathParts.length >= 2) {
        return `${pathParts[0]}/${pathParts[1]}`;
      }
    }
    return input;
  };

  const handleFetchRepo = async () => {
    const parsedRepoUrl = parseRepoUrl(repoUrl);
    setIsLoading(true);
    setIsLandingPage(false);
    setError(null);

    try {
      const response = await fetch('/api/analyze_repo', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
        body: JSON.stringify({ repo_url: parsedRepoUrl }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data: EnhancedRepoData = await response.json();
      setRepoData(data);
    } catch (error) {
      console.error('Error fetching repo data:', error);
      setError('Error fetching repository data. Please check the URL and try again.');
      setIsLandingPage(true);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      handleFetchRepo();
    }
  };

  // Transform commit data for visualization
  const commitData = repoData ? Object.entries(repoData.monthly_commits)
    .map(([date, commits]) => ({
      month: new Date(date).toLocaleString('default', { month: 'short' }),
      commits,
    }))
    .slice(-12)
    .reverse() : [];

  // Transform issue data for charts
  const issuesBySeverity = repoData ? [
    { name: 'Critical', value: repoData.issues.statistics.critical, color: SEVERITY_COLORS.critical },
    { name: 'High', value: repoData.issues.statistics.high, color: SEVERITY_COLORS.high },
    { name: 'Medium', value: repoData.issues.statistics.medium, color: SEVERITY_COLORS.medium },
    { name: 'Low', value: repoData.issues.statistics.low, color: SEVERITY_COLORS.low },
  ].filter(item => item.value > 0) : [];

  const issuesByCategory = repoData ? Object.entries(repoData.issues.statistics.by_category)
    .map(([category, count]) => ({
      name: category.charAt(0).toUpperCase() + category.slice(1),
      value: count,
      color: CATEGORY_COLORS[category as keyof typeof CATEGORY_COLORS] || '#6b7280'
    }))
    .filter(item => item.value > 0) : [];

  // Complexity distribution data
  const complexityData = repoData ? repoData.visualizations.complexity_heatmap.files
    .map(file => ({
      name: file.file.split('/').pop() || file.file,
      complexity: file.total_complexity,
      lines: file.lines,
      risk: file.risk_level
    }))
    .sort((a, b) => b.complexity - a.complexity)
    .slice(0, 20) : [];

  if (isLandingPage) {
    return (
      <div className="min-h-screen bg-background text-foreground">
        <div className="flex flex-col items-center justify-center min-h-screen p-4">
          <div className="text-center mb-8">
            <h1 className="text-4xl font-bold flex items-center justify-center gap-3 mb-4">
              <img src="cg.png" alt="CG Logo" className="h-12 w-12" />
              <span>Enhanced Codebase Analytics</span>
            </h1>
            <p className="text-muted-foreground">Comprehensive code quality analysis with rich visualizations and error detection using graph-sitter</p>
          </div>
          <div className="flex items-center gap-3 w-full max-w-lg">
            <Input
              type="text"
              placeholder="Enter the GitHub repo link or owner/repo"
              value={repoUrl}
              onChange={(e) => setRepoUrl(e.target.value)}
              onKeyPress={handleKeyPress}
              className="flex-1"
              title="Format: https://github.com/owner/repo or owner/repo"
              aria-label="Repository URL input"
            />
            <Button
              onClick={handleFetchRepo}
              disabled={isLoading}
              aria-label={isLoading ? "Analyzing repository" : "Analyze repository"}
            >
              {isLoading ? "Analyzing..." : "Analyze"}
            </Button>
          </div>
          {isLoading && (
            <div className="mt-8 text-center" role="status" aria-live="polite">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto mb-4" aria-hidden="true"></div>
              <p className="text-sm text-muted-foreground">Performing comprehensive code analysis with graph-sitter...</p>
            </div>
          )}
          {error && (
            <Alert variant="destructive" className="mt-4 max-w-lg" role="alert">
              <AlertTriangle className="h-4 w-4" />
              <AlertTitle>Error</AlertTitle>
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}
        </div>
      </div>
    );
  }

  if (!repoData) return null;

  return (
    <div className="min-h-screen bg-background text-foreground p-6">
      {/* Header */}
      <header className="mb-8">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h1 className="text-3xl font-bold flex items-center gap-2">
              <Github className="h-8 w-8" aria-hidden="true" />
              {repoData.repo_url}
            </h1>
            <p className="text-muted-foreground mt-1">{repoData.description}</p>
          </div>
          <div className="flex items-center gap-4">
            <div className="text-center">
              <div className="text-2xl font-bold" aria-label={`Quality score: ${repoData.quality_score} out of 100`}>{repoData.quality_score}</div>
              <div className="text-sm text-muted-foreground">Quality Score</div>
            </div>
            <Badge 
              variant={repoData.quality_grade === 'A' ? 'default' : repoData.quality_grade === 'B' ? 'secondary' : 'destructive'} 
              className="text-lg px-3 py-1"
              aria-label={`Quality grade: ${repoData.quality_grade}`}
            >
              Grade {repoData.quality_grade}
            </Badge>
          </div>
        </div>
      </header>

      <Tabs defaultValue="overview" className="space-y-4">
        <TabsList className="grid grid-cols-1 md:grid-cols-5 lg:grid-cols-6 h-auto gap-2">
          <TabsTrigger value="overview" className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground">Overview</TabsTrigger>
          <TabsTrigger value="issues" className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground">Issues</TabsTrigger>
          <TabsTrigger value="complexity" className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground">Complexity</TabsTrigger>
          <TabsTrigger value="dependencies" className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground">Dependencies</TabsTrigger>
          <TabsTrigger value="trends" className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground">Trends</TabsTrigger>
          <TabsTrigger value="upgrade" className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground">Upgrade</TabsTrigger>
        </TabsList>

        {/* Overview Tab */}
        <TabsContent value="overview" className="space-y-6" role="tabpanel" id="overview-panel" aria-labelledby="overview-tab">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {/* Quality Metrics */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Brain className="h-5 w-5" aria-hidden="true" />
                  Quality Metrics
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span>Maintainability</span>
                    <span aria-label={`Maintainability: ${repoData.maintainability_index.average} percent`}>{repoData.maintainability_index.average}%</span>
                  </div>
                  <Progress value={repoData.maintainability_index.average} className="h-2" aria-label="Maintainability progress" />
                </div>
                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span>Code Quality</span>
                    <span aria-label={`Code quality: ${repoData.quality_score} percent`}>{repoData.quality_score}%</span>
                  </div>
                  <Progress value={repoData.quality_score} className="h-2" aria-label="Code quality progress" />
                </div>
                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span>Comment Density</span>
                    <span aria-label={`Comment density: ${repoData.line_metrics.total.comment_density.toFixed(1)} percent`}>{repoData.line_metrics.total.comment_density.toFixed(1)}%</span>
                  </div>
                  <Progress value={repoData.line_metrics.total.comment_density} className="h-2" aria-label="Comment density progress" />
                </div>
              </CardContent>
            </Card>

            {/* Risk Distribution */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Shield className="h-5 w-5" aria-hidden="true" />
                  Risk Distribution
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3" role="list" aria-label="Risk distribution breakdown">
                  <div className="flex justify-between items-center" role="listitem">
                    <span className="text-sm">High Risk Files</span>
                    <Badge variant="destructive" aria-label={`${repoData.visualizations.risk_distribution.high_risk_files} high risk files`}>
                      {repoData.visualizations.risk_distribution.high_risk_files}
                    </Badge>
                  </div>
                  <div className="flex justify-between items-center" role="listitem">
                    <span className="text-sm">Medium Risk Files</span>
                    <Badge variant="secondary" aria-label={`${repoData.visualizations.risk_distribution.medium_risk_files} medium risk files`}>
                      {repoData.visualizations.risk_distribution.medium_risk_files}
                    </Badge>
                  </div>
                  <div className="flex justify-between items-center" role="listitem">
                    <span className="text-sm">Low Risk Files</span>
                    <Badge variant="outline" aria-label={`${repoData.visualizations.risk_distribution.low_risk_files} low risk files`}>
                      {repoData.visualizations.risk_distribution.low_risk_files}
                    </Badge>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Issues by Severity */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <AlertTriangle className="h-5 w-5" aria-hidden="true" />
                  Issues by Severity
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div aria-label="Issues by severity chart">
                  <ResponsiveContainer width="100%" height={200}>
                    <PieChart>
                      <Pie
                        data={issuesBySeverity}
                        cx="50%"
                        cy="50%"
                        innerRadius={40}
                        outerRadius={80}
                        dataKey="value"
                        aria-label="Issues by severity pie chart"
                      >
                        {issuesBySeverity.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={entry.color} />
                        ))}
                      </Pie>
                      <Tooltip />
                      <Legend />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Commit Activity */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <GitBranch className="h-5 w-5" aria-hidden="true" />
                Commit Activity (Last 12 Months)
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div aria-label="Commit activity chart for the last 12 months">
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={commitData}>
                    <XAxis dataKey="month" />
                    <YAxis />
                    <Tooltip />
                    <Bar dataKey="commits" fill="#3b82f6" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Issues & Errors Tab */}
        <TabsContent value="issues" className="space-y-6" role="tabpanel" id="issues-panel" aria-labelledby="issues-tab">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
            {/* Issues by Category */}
            <Card>
              <CardHeader>
                <CardTitle>Issues by Category</CardTitle>
              </CardHeader>
              <CardContent>
                <div aria-label="Issues by category chart">
                  <ResponsiveContainer width="100%" height={250}>
                    <BarChart data={issuesByCategory}>
                      <XAxis dataKey="name" />
                      <YAxis />
                      <Tooltip />
                      <Bar dataKey="value" fill="#8884d8" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>

            {/* Critical Issues Alert */}
            {repoData.issues.statistics.critical > 0 && (
              <Card>
                <CardHeader>
                  <CardTitle className="text-red-600">Critical Issues Detected</CardTitle>
                </CardHeader>
                <CardContent>
                  <Alert variant="destructive" role="alert">
                    <AlertTriangle className="h-4 w-4" />
                    <AlertTitle>Immediate Attention Required</AlertTitle>
                    <AlertDescription>
                      {repoData.issues.statistics.critical} critical security or quality issues found that require immediate attention.
                    </AlertDescription>
                  </Alert>
                </CardContent>
              </Card>
            )}
          </div>

          {/* Issues List */}
          <Card>
            <CardHeader>
              <CardTitle>Detailed Issues</CardTitle>
              <CardDescription>
                Showing {repoData.issues.details.length} of {repoData.issues.statistics.total} total issues
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4 max-h-96 overflow-y-auto" role="list" aria-label="List of code issues">
                {repoData.issues.details.map((issue, index) => (
                  <div 
                    key={index} 
                    className="border rounded-lg p-4 hover:bg-muted/50 cursor-pointer focus:outline-none focus:ring-2 focus:ring-ring" 
                    onClick={() => setSelectedIssue(issue)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' || e.key === ' ') {
                        e.preventDefault();
                        setSelectedIssue(issue);
                      }
                    }}
                    tabIndex={0}
                    role="listitem"
                    aria-label={`Issue: ${issue.type} in ${issue.file_path}`}
                  >
                    <div className="flex items-start justify-between mb-2">
                      <div className="flex items-center gap-2">
                        <Badge 
                          variant={issue.severity === 'critical' ? 'destructive' : issue.severity === 'high' ? 'destructive' : 'secondary'}
                          aria-label={`Severity: ${issue.severity}`}
                        >
                          {issue.severity}
                        </Badge>
                        <Badge variant="outline" aria-label={`Category: ${issue.category}`}>{issue.category}</Badge>
                      </div>
                      <span className="text-sm text-muted-foreground" aria-label={`File: ${issue.file_path}`}>{issue.file_path}</span>
                    </div>
                    <h4 className="font-medium mb-1">{issue.type}</h4>
                    <p className="text-sm text-muted-foreground mb-2">{issue.message}</p>
                    <p className="text-sm text-blue-600">{issue.suggestion}</p>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Complexity Tab */}
        <TabsContent value="complexity" className="space-y-6" role="tabpanel" id="complexity-panel" aria-labelledby="complexity-tab">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <TrendingUp className="h-5 w-5" aria-hidden="true" />
                Complexity Heatmap
              </CardTitle>
              <CardDescription>Files with highest complexity (top 20)</CardDescription>
            </CardHeader>
            <CardContent>
              <div aria-label="Complexity vs lines of code scatter plot">
                <ResponsiveContainer width="100%" height={400}>
                  <ScatterChart data={complexityData}>
                    <XAxis dataKey="lines" name="Lines of Code" />
                    <YAxis dataKey="complexity" name="Complexity" />
                    <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                    <Scatter dataKey="complexity" fill="#8884d8" />
                  </ScatterChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>

          {/* File Metrics Table */}
          <Card>
            <CardHeader>
              <CardTitle>File Metrics</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full text-sm" role="table" aria-label="File metrics table">
                  <thead>
                    <tr className="border-b" role="row">
                      <th className="text-left p-2" role="columnheader">File</th>
                      <th className="text-right p-2" role="columnheader">LOC</th>
                      <th className="text-right p-2" role="columnheader">Complexity</th>
                      <th className="text-right p-2" role="columnheader">Maintainability</th>
                      <th className="text-right p-2" role="columnheader">Risk Score</th>
                    </tr>
                  </thead>
                  <tbody>
                    {repoData.visualizations.file_metrics.slice(0, 20).map((file, index) => (
                      <tr key={index} className="border-b hover:bg-muted/50" role="row">
                        <td className="p-2 font-mono text-sm" role="cell">{file.name}</td>
                        <td className="p-2 text-right" role="cell">{file.loc}</td>
                        <td className="p-2 text-right" role="cell">{file.complexity.toFixed(1)}</td>
                        <td className="p-2 text-right" role="cell">{file.maintainability.toFixed(1)}%</td>
                        <td className="p-2 text-right" role="cell">
                          <Badge 
                            variant={file.risk_score > 70 ? 'destructive' : file.risk_score > 30 ? 'secondary' : 'outline'}
                            aria-label={`Risk score: ${file.risk_score.toFixed(1)}`}
                          >
                            {file.risk_score.toFixed(1)}
                          </Badge>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Dependencies Tab */}
        <TabsContent value="dependencies" className="space-y-6" role="tabpanel" id="dependencies-panel" aria-labelledby="dependencies-tab">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Network className="h-5 w-5" aria-hidden="true" />
                Dependency Overview
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4" role="region" aria-label="Dependency statistics">
                <div className="text-center">
                  <div className="text-2xl font-bold" aria-label={`${repoData.visualizations.dependency_graph.stats.total_nodes} total nodes`}>
                    {repoData.visualizations.dependency_graph.stats.total_nodes}
                  </div>
                  <div className="text-sm text-muted-foreground">Total Nodes</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold" aria-label={`${repoData.visualizations.dependency_graph.stats.total_edges} dependencies`}>
                    {repoData.visualizations.dependency_graph.stats.total_edges}
                  </div>
                  <div className="text-sm text-muted-foreground">Dependencies</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold" aria-label={`${repoData.visualizations.dependency_graph.stats.file_count} files`}>
                    {repoData.visualizations.dependency_graph.stats.file_count}
                  </div>
                  <div className="text-sm text-muted-foreground">Files</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold" aria-label={`${repoData.visualizations.dependency_graph.stats.function_count} functions`}>
                    {repoData.visualizations.dependency_graph.stats.function_count}
                  </div>
                  <div className="text-sm text-muted-foreground">Functions</div>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Dependency Graph</CardTitle>
              <CardDescription>Interactive visualization powered by graph-sitter analysis</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-96 bg-muted/20 rounded-lg flex items-center justify-center" role="img" aria-label="Dependency graph visualization placeholder">
                <div className="text-center">
                  <Network className="h-16 w-16 mx-auto mb-4 text-muted-foreground" aria-hidden="true" />
                  <p className="text-muted-foreground">Interactive dependency graph visualization</p>
                  <p className="text-sm text-muted-foreground mt-2">
                    {repoData.visualizations.dependency_graph.nodes.length} nodes, {repoData.visualizations.dependency_graph.edges.length} edges
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Trends Tab */}
        <TabsContent value="trends" className="space-y-6" role="tabpanel" id="trends-panel" aria-labelledby="trends-tab">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <TrendingUp className="h-5 w-5" aria-hidden="true" />
                Development Activity
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div aria-label="Development activity trend line chart">
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={commitData}>
                    <XAxis dataKey="month" />
                    <YAxis />
                    <Tooltip />
                    <Line type="monotone" dataKey="commits" stroke="#3b82f6" strokeWidth={2} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>Quality Trends</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4" role="list" aria-label="Quality metrics">
                  <div className="flex justify-between items-center" role="listitem">
                    <span>Overall Quality</span>
                    <div className="flex items-center gap-2">
                      <Progress value={repoData.quality_score} className="w-24 h-2" aria-label="Overall quality progress" />
                      <span className="text-sm font-medium" aria-label={`Overall quality: ${repoData.quality_score} percent`}>{repoData.quality_score}%</span>
                    </div>
                  </div>
                  <div className="flex justify-between items-center" role="listitem">
                    <span>Maintainability</span>
                    <div className="flex items-center gap-2">
                      <Progress value={repoData.maintainability_index.average} className="w-24 h-2" aria-label="Maintainability progress" />
                      <span className="text-sm font-medium" aria-label={`Maintainability: ${repoData.maintainability_index.average} percent`}>{repoData.maintainability_index.average}%</span>
                    </div>
                  </div>
                  <div className="flex justify-between items-center" role="listitem">
                    <span>Code Coverage</span>
                    <div className="flex items-center gap-2">
                      <Progress value={repoData.line_metrics.total.comment_density} className="w-24 h-2" aria-label="Code coverage progress" />
                      <span className="text-sm font-medium" aria-label={`Code coverage: ${repoData.line_metrics.total.comment_density.toFixed(1)} percent`}>{repoData.line_metrics.total.comment_density.toFixed(1)}%</span>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Recommendations</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3" role="list" aria-label="Improvement recommendations">
                  {repoData.issues.statistics.critical > 0 && (
                    <Alert variant="destructive" role="listitem">
                      <AlertTriangle className="h-4 w-4" />
                      <AlertDescription>
                        Address {repoData.issues.statistics.critical} critical security issues immediately
                      </AlertDescription>
                    </Alert>
                  )}
                  {repoData.cyclomatic_complexity.average > 15 && (
                    <Alert role="listitem">
                      <AlertDescription>
                        Consider refactoring complex functions to improve maintainability
                      </AlertDescription>
                    </Alert>
                  )}
                  {repoData.line_metrics.total.comment_density < 10 && (
                    <Alert role="listitem">
                      <AlertDescription>
                        Increase code documentation to improve maintainability
                      </AlertDescription>
                    </Alert>
                  )}
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* Upgrade Tab */}
        <TabsContent value="upgrade" className="space-y-6" role="tabpanel" id="upgrade-panel" aria-labelledby="upgrade-tab">
          <UpgradeRunUI repoUrl={repoData.repo_url} />
        </TabsContent>
      </Tabs>
    </div>
  );
}
