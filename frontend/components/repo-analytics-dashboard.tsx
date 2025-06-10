"use client"

import React, { useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible"
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  LineChart, Line, PieChart, Pie, Cell, Area, AreaChart
} from 'recharts'
import { 
  GitBranch, FileText, Code, AlertTriangle, CheckCircle, 
  XCircle, Info, ChevronDown, ChevronRight, Folder, 
  FolderOpen, File, Activity, TrendingUp, Users, Calendar,
  Zap, Shield, Target, Layers, Database, Cloud, Settings,
  Download, Share, RefreshCw, Eye, BarChart3
} from 'lucide-react'

// Types
interface RepositoryNode {
  name: string
  path: string
  type: 'file' | 'directory'
  issue_count: number
  critical_issues: number
  functional_issues: number
  minor_issues: number
  children?: RepositoryNode[]
  issues?: IssueItem[]
}

interface IssueItem {
  file_path: string
  line_number: number
  severity: 'critical' | 'functional' | 'minor'
  issue_type: string
  description: string
  suggestion?: string
}

interface AnalysisResult {
  repo_url: string
  description: string
  basic_metrics: {
    files: number
    functions: number
    classes: number
    modules: number
  }
  line_metrics: {
    total: {
      loc: number
      lloc: number
      sloc: number
      comments: number
      comment_density: number
    }
  }
  complexity_metrics: {
    cyclomatic_complexity: { average: number }
    maintainability_index: { average: number }
    halstead_metrics: { total_volume: number; average_volume: number }
  }
  repository_structure: RepositoryNode
  issues_summary: {
    total: number
    critical: number
    functional: number
    minor: number
  }
  detailed_issues: IssueItem[]
  monthly_commits: Record<string, number>
}

interface DeploymentConfig {
  mode: 'local' | 'modal' | 'docker'
  endpoint: string
  status: 'idle' | 'deploying' | 'deployed' | 'error'
}

// Repository Tree Component
const RepositoryTree: React.FC<{ 
  node: RepositoryNode; 
  level?: number;
}> = ({ node, level = 0 }) => {
  const [isExpanded, setIsExpanded] = useState(level < 2)
  
  const hasIssues = node.issue_count > 0
  const isDirectory = node.type === 'directory'

  return (
    <div className="space-y-1">
      <div 
        className={`flex items-center space-x-2 p-2 rounded-md hover:bg-gray-50 cursor-pointer transition-colors ${
          hasIssues ? 'border-l-4 border-l-red-200 bg-red-50/30' : ''
        }`}
        style={{ paddingLeft: `${level * 20 + 8}px` }}
        onClick={() => {
          if (isDirectory) {
            setIsExpanded(!isExpanded)
          }
        }}
      >
        {isDirectory && (
          <span className="text-gray-400 transition-transform">
            {isExpanded ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
          </span>
        )}
        
        <span className="text-gray-500">
          {isDirectory ? (
            isExpanded ? <FolderOpen size={16} /> : <Folder size={16} />
          ) : (
            <File size={16} />
          )}
        </span>
        
        <span className="font-medium text-gray-900 flex-1">{node.name}</span>
        
        {hasIssues && (
          <div className="flex space-x-1">
            {node.critical_issues > 0 && (
              <Badge variant="destructive" className="text-xs">
                {node.critical_issues}
              </Badge>
            )}
            {node.functional_issues > 0 && (
              <Badge className="text-xs bg-yellow-100 text-yellow-800">
                {node.functional_issues}
              </Badge>
            )}
            {node.minor_issues > 0 && (
              <Badge variant="outline" className="text-xs">
                {node.minor_issues}
              </Badge>
            )}
          </div>
        )}
      </div>

      {isDirectory && isExpanded && node.children && (
        <div>
          {node.children.map((child, index) => (
            <RepositoryTree 
              key={index} 
              node={child} 
              level={level + 1}
            />
          ))}
        </div>
      )}
    </div>
  )
}

// Deployment Status Component
const DeploymentStatus: React.FC<{
  config: DeploymentConfig;
  onDeploy: (mode: 'local' | 'modal' | 'docker') => void;
}> = ({ config, onDeploy }) => {
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'deployed': return 'text-green-600 bg-green-50'
      case 'deploying': return 'text-yellow-600 bg-yellow-50'
      case 'error': return 'text-red-600 bg-red-50'
      default: return 'text-gray-600 bg-gray-50'
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center space-x-2">
          <Cloud size={20} />
          <span>Deployment Status</span>
        </CardTitle>
        <CardDescription>
          Current deployment configuration and status
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <div className="font-medium">Mode: {config.mode}</div>
            <div className="text-sm text-gray-600">Endpoint: {config.endpoint}</div>
          </div>
          <Badge className={getStatusColor(config.status)}>
            {config.status}
          </Badge>
        </div>
        
        <div className="grid grid-cols-3 gap-2">
          <Button 
            variant={config.mode === 'local' ? 'default' : 'outline'}
            size="sm"
            onClick={() => onDeploy('local')}
            disabled={config.status === 'deploying'}
          >
            <Settings size={16} className="mr-1" />
            Local
          </Button>
          <Button 
            variant={config.mode === 'modal' ? 'default' : 'outline'}
            size="sm"
            onClick={() => onDeploy('modal')}
            disabled={config.status === 'deploying'}
          >
            <Zap size={16} className="mr-1" />
            Modal
          </Button>
          <Button 
            variant={config.mode === 'docker' ? 'default' : 'outline'}
            size="sm"
            onClick={() => onDeploy('docker')}
            disabled={config.status === 'deploying'}
          >
            <Database size={16} className="mr-1" />
            Docker
          </Button>
        </div>
        
        {config.status === 'deploying' && (
          <div className="space-y-2">
            <Progress value={66} className="w-full" />
            <div className="text-sm text-gray-600">Deploying {config.mode} environment...</div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}

// Main Dashboard Component
export default function RepoAnalyticsDashboard() {
  const [repoUrl, setRepoUrl] = useState('')
  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [deploymentConfig, setDeploymentConfig] = useState<DeploymentConfig>({
    mode: 'local',
    endpoint: 'http://localhost:8000',
    status: 'idle'
  })

  const analyzeRepository = async () => {
    if (!repoUrl.trim()) {
      setError('Please enter a repository URL')
      return
    }

    setLoading(true)
    setError('')
    
    try {
      const response = await fetch(`${deploymentConfig.endpoint}/analyze_repo`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ repo_url: repoUrl }),
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'Analysis failed')
      }

      const result = await response.json()
      setAnalysis(result)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred')
    } finally {
      setLoading(false)
    }
  }

  const handleDeploy = async (mode: 'local' | 'modal' | 'docker') => {
    setDeploymentConfig(prev => ({ ...prev, status: 'deploying', mode }))
    
    // Simulate deployment process
    setTimeout(() => {
      const endpoints = {
        local: 'http://localhost:8000',
        modal: 'https://your-modal-app.modal.run',
        docker: 'http://localhost:80/api'
      }
      
      setDeploymentConfig({
        mode,
        endpoint: endpoints[mode],
        status: 'deployed'
      })
    }, 3000)
  }

  // Prepare chart data
  const issueDistributionData = analysis ? [
    { name: 'Critical', value: analysis.issues_summary.critical, color: '#ef4444' },
    { name: 'Functional', value: analysis.issues_summary.functional, color: '#f59e0b' },
    { name: 'Minor', value: analysis.issues_summary.minor, color: '#3b82f6' },
  ] : []

  const commitData = analysis ? Object.entries(analysis.monthly_commits).map(([month, commits]) => ({
    month,
    commits
  })) : []

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 p-6">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <div className="text-center space-y-4">
          <h1 className="text-5xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
            🚀 Enhanced Codebase Analytics
          </h1>
          <p className="text-xl text-gray-600">
            Comprehensive repository analysis with advanced deployment options
          </p>
        </div>

        {/* Deployment Status */}
        <DeploymentStatus 
          config={deploymentConfig}
          onDeploy={handleDeploy}
        />

        {/* Input Section */}
        <Card className="border-2 border-dashed border-gray-300 hover:border-blue-400 transition-colors">
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <GitBranch className="h-5 w-5" />
              <span>Repository Analysis</span>
            </CardTitle>
            <CardDescription>
              Enter a GitHub repository URL to analyze code quality, structure, and issues
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex space-x-2">
              <Input
                placeholder="https://github.com/owner/repository"
                value={repoUrl}
                onChange={(e) => setRepoUrl(e.target.value)}
                className="flex-1"
              />
              <Button 
                onClick={analyzeRepository} 
                disabled={loading || deploymentConfig.status === 'deploying'}
                className="px-6"
              >
                {loading ? (
                  <>
                    <RefreshCw className="mr-2 h-4 w-4 animate-spin" />
                    Analyzing...
                  </>
                ) : (
                  <>
                    <BarChart3 className="mr-2 h-4 w-4" />
                    Analyze Repository
                  </>
                )}
              </Button>
            </div>
            
            {error && (
              <Alert variant="destructive">
                <XCircle className="h-4 w-4" />
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}
            
            {deploymentConfig.status === 'deployed' && (
              <Alert>
                <CheckCircle className="h-4 w-4" />
                <AlertDescription>
                  Connected to {deploymentConfig.mode} deployment at {deploymentConfig.endpoint}
                </AlertDescription>
              </Alert>
            )}
          </CardContent>
        </Card>

        {/* Results */}
        {analysis && (
          <div className="space-y-6">
            {/* Overview Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              <Card className="bg-gradient-to-br from-blue-50 to-blue-100 border-blue-200">
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-medium text-blue-700">Total Files</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-3xl font-bold text-blue-900">{analysis.basic_metrics.files}</div>
                  <div className="text-xs text-blue-600 mt-1">
                    {analysis.basic_metrics.functions} functions, {analysis.basic_metrics.classes} classes
                  </div>
                </CardContent>
              </Card>

              <Card className="bg-gradient-to-br from-green-50 to-green-100 border-green-200">
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-medium text-green-700">Lines of Code</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-3xl font-bold text-green-900">{analysis.line_metrics.total.loc.toLocaleString()}</div>
                  <div className="text-xs text-green-600 mt-1">
                    {(analysis.line_metrics.total.comment_density * 100).toFixed(1)}% comments
                  </div>
                </CardContent>
              </Card>

              <Card className="bg-gradient-to-br from-purple-50 to-purple-100 border-purple-200">
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-medium text-purple-700">Code Quality</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-3xl font-bold text-purple-900">
                    {analysis.complexity_metrics.maintainability_index.average.toFixed(0)}
                  </div>
                  <div className="text-xs text-purple-600 mt-1">Maintainability Index</div>
                  <Progress 
                    value={analysis.complexity_metrics.maintainability_index.average} 
                    className="mt-2"
                  />
                </CardContent>
              </Card>

              <Card className="bg-gradient-to-br from-red-50 to-red-100 border-red-200">
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-medium text-red-700">Issues Found</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-3xl font-bold text-red-900">{analysis.issues_summary.total}</div>
                  <div className="text-xs text-red-600 mt-1">
                    {analysis.issues_summary.critical} critical, {analysis.issues_summary.functional} functional
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Analysis Tabs */}
            <Tabs defaultValue="structure" className="space-y-4">
              <TabsList className="grid w-full grid-cols-4">
                <TabsTrigger value="structure">Repository Structure</TabsTrigger>
                <TabsTrigger value="metrics">Metrics</TabsTrigger>
                <TabsTrigger value="issues">Issues</TabsTrigger>
                <TabsTrigger value="activity">Activity</TabsTrigger>
              </TabsList>

              <TabsContent value="structure" className="space-y-4">
                <Card>
                  <CardHeader>
                    <CardTitle>Repository Tree</CardTitle>
                    <CardDescription>
                      Navigate through your repository structure and view issues by file
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="max-h-96 overflow-y-auto border rounded-md p-4 bg-white">
                      <RepositoryTree node={analysis.repository_structure} />
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>

              <TabsContent value="metrics" className="space-y-4">
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                  <Card>
                    <CardHeader>
                      <CardTitle>Issue Distribution</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <ResponsiveContainer width="100%" height={250}>
                        <PieChart>
                          <Pie
                            data={issueDistributionData}
                            cx="50%"
                            cy="50%"
                            outerRadius={80}
                            dataKey="value"
                            label={({ name, value }) => `${name}: ${value}`}
                          >
                            {issueDistributionData.map((entry, index) => (
                              <Cell key={`cell-${index}`} fill={entry.color} />
                            ))}
                          </Pie>
                          <Tooltip />
                        </PieChart>
                      </ResponsiveContainer>
                    </CardContent>
                  </Card>

                  <Card>
                    <CardHeader>
                      <CardTitle>Complexity Metrics</CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <div className="flex justify-between items-center">
                        <span className="text-sm">Cyclomatic Complexity</span>
                        <Badge variant="outline">
                          {analysis.complexity_metrics.cyclomatic_complexity.average.toFixed(1)}
                        </Badge>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-sm">Maintainability Index</span>
                        <Badge variant="outline">
                          {analysis.complexity_metrics.maintainability_index.average.toFixed(0)}/100
                        </Badge>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-sm">Halstead Volume</span>
                        <Badge variant="outline">
                          {analysis.complexity_metrics.halstead_metrics.total_volume}
                        </Badge>
                      </div>
                    </CardContent>
                  </Card>
                </div>
              </TabsContent>

              <TabsContent value="issues" className="space-y-4">
                <Card>
                  <CardHeader>
                    <CardTitle>Detailed Issues Report</CardTitle>
                    <CardDescription>
                      Critical issues that need immediate attention
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    {analysis.detailed_issues.slice(0, 10).map((issue, index) => (
                      <Alert key={index} className={
                        issue.severity === 'critical' ? 'border-red-200 bg-red-50' :
                        issue.severity === 'functional' ? 'border-yellow-200 bg-yellow-50' :
                        'border-blue-200 bg-blue-50'
                      }>
                        <AlertTriangle className="h-4 w-4" />
                        <AlertDescription>
                          <div className="flex justify-between items-start">
                            <div>
                              <div className="font-medium">
                                {issue.file_path}:{issue.line_number}
                              </div>
                              <div className="text-sm mt-1">{issue.description}</div>
                              {issue.suggestion && (
                                <div className="text-xs mt-2 opacity-75">
                                  💡 {issue.suggestion}
                                </div>
                              )}
                            </div>
                            <Badge variant={
                              issue.severity === 'critical' ? 'destructive' :
                              issue.severity === 'functional' ? 'secondary' : 'outline'
                            }>
                              {issue.severity}
                            </Badge>
                          </div>
                        </AlertDescription>
                      </Alert>
                    ))}
                  </CardContent>
                </Card>
              </TabsContent>

              <TabsContent value="activity" className="space-y-4">
                <Card>
                  <CardHeader>
                    <CardTitle>Git Commit Activity</CardTitle>
                    <CardDescription>
                      Repository activity over the last 12 months
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <ResponsiveContainer width="100%" height={300}>
                      <AreaChart data={commitData}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="month" />
                        <YAxis />
                        <Tooltip />
                        <Area type="monotone" dataKey="commits" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.3} />
                      </AreaChart>
                    </ResponsiveContainer>
                  </CardContent>
                </Card>
              </TabsContent>
            </Tabs>
          </div>
        )}
      </div>
    </div>
  )
}

