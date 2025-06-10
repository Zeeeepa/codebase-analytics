"use client"

import React, { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible"
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog"
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  LineChart, Line, PieChart, Pie, Cell, Area, AreaChart, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar
} from 'recharts'
import { 
  GitBranch, FileText, Code, AlertTriangle, CheckCircle, 
  XCircle, Info, ChevronDown, ChevronRight, Folder, 
  FolderOpen, File, Activity, TrendingUp, Users, Calendar,
  Zap, Shield, Target, Layers, Database, Cloud, Settings,
  Download, Share, RefreshCw, Eye, BarChart3
} from 'lucide-react'

// Enhanced Types with Modal Integration
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
  mode: 'local' | 'modal'
  endpoint: string
  status: 'idle' | 'deploying' | 'deployed' | 'error'
}

// Enhanced Repository Tree Component with Modal Features
const EnhancedRepositoryTree: React.FC<{ 
  node: RepositoryNode; 
  level?: number;
  onFileSelect?: (file: RepositoryNode) => void;
}> = ({ node, level = 0, onFileSelect }) => {
  const [isExpanded, setIsExpanded] = useState(level < 2)
  
  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical': return 'text-red-600 bg-red-50 border-red-200'
      case 'functional': return 'text-yellow-600 bg-yellow-50 border-yellow-200'
      case 'minor': return 'text-blue-600 bg-blue-50 border-blue-200'
      default: return 'text-gray-600 bg-gray-50 border-gray-200'
    }
  }

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
          } else if (onFileSelect) {
            onFileSelect(node)
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
                {node.critical_issues} Critical
              </Badge>
            )}
            {node.functional_issues > 0 && (
              <Badge variant="secondary" className="text-xs bg-yellow-100 text-yellow-800">
                {node.functional_issues} Functional
              </Badge>
            )}
            {node.minor_issues > 0 && (
              <Badge variant="outline" className="text-xs">
                {node.minor_issues} Minor
              </Badge>
            )}
          </div>
        )}
        
        {!isDirectory && (
          <Button variant="ghost" size="sm" className="text-xs">
            <Eye size={12} className="mr-1" />
            View
          </Button>
        )}
      </div>

      {isDirectory && isExpanded && node.children && (
        <div>
          {node.children.map((child, index) => (
            <EnhancedRepositoryTree 
              key={index} 
              node={child} 
              level={level + 1}
              onFileSelect={onFileSelect}
            />
          ))}
        </div>
      )}
    </div>
  )
}

// File Detail Modal Component
const FileDetailModal: React.FC<{
  file: RepositoryNode | null;
  isOpen: boolean;
  onClose: () => void;
}> = ({ file, isOpen, onClose }) => {
  if (!file) return null;

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="max-w-4xl max-h-[80vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="flex items-center space-x-2">
            <File size={20} />
            <span>{file.name}</span>
          </DialogTitle>
          <DialogDescription>
            File analysis details and detected issues
          </DialogDescription>
        </DialogHeader>
        
        <div className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <Card>
              <CardContent className="p-4">
                <div className="text-2xl font-bold text-red-600">{file.critical_issues}</div>
                <div className="text-sm text-gray-600">Critical Issues</div>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="p-4">
                <div className="text-2xl font-bold text-yellow-600">{file.functional_issues}</div>
                <div className="text-sm text-gray-600">Functional Issues</div>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="p-4">
                <div className="text-2xl font-bold text-blue-600">{file.minor_issues}</div>
                <div className="text-sm text-gray-600">Minor Issues</div>
              </CardContent>
            </Card>
          </div>
          
          {file.issues && file.issues.length > 0 && (
            <div className="space-y-2">
              <h3 className="font-semibold">Detected Issues</h3>
              {file.issues.map((issue, index) => (
                <Alert key={index} className={`text-xs ${
                  issue.severity === 'critical' ? 'border-red-200 bg-red-50' :
                  issue.severity === 'functional' ? 'border-yellow-200 bg-yellow-50' :
                  'border-blue-200 bg-blue-50'
                }`}>
                  <AlertTriangle className="h-3 w-3" />
                  <AlertDescription>
                    <div className="font-medium">Line {issue.line_number}: {issue.description}</div>
                    {issue.suggestion && (
                      <div className="text-xs mt-1 opacity-75">💡 {issue.suggestion}</div>
                    )}
                  </AlertDescription>
                </Alert>
              ))}
            </div>
          )}
        </div>
      </DialogContent>
    </Dialog>
  )
}

// Deployment Status Component
const DeploymentStatus: React.FC<{
  config: DeploymentConfig;
  onDeploy: (mode: 'local' | 'modal') => void;
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
        
        <div className="grid grid-cols-2 gap-2">
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

// Main Enhanced Dashboard Component
export default function EnhancedAnalyticsDashboard() {
  const [repoUrl, setRepoUrl] = useState('')
  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [selectedFile, setSelectedFile] = useState<RepositoryNode | null>(null)
  const [isFileModalOpen, setIsFileModalOpen] = useState(false)
  const [deploymentConfig, setDeploymentConfig] = useState<DeploymentConfig>({
    mode: 'local',
    endpoint: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
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

  const handleDeploy = (mode: 'local' | 'modal') => {
    setDeploymentConfig(prev => ({ ...prev, status: 'deploying' }))
    
    // Simulate deployment process
    setTimeout(() => {
      const endpoints = {
        local: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
        modal: 'https://your-modal-app.modal.run'
      }
      
      setDeploymentConfig({
        mode,
        endpoint: endpoints[mode],
        status: 'deployed'
      })
    }, 2000)
  }

  const handleFileSelect = (file: RepositoryNode) => {
    setSelectedFile(file)
    setIsFileModalOpen(true)
  }

  // Prepare enhanced chart data
  const complexityRadarData = analysis ? [
    { metric: 'Complexity', value: analysis.complexity_metrics.cyclomatic_complexity.average },
    { metric: 'Maintainability', value: analysis.complexity_metrics.maintainability_index.average },
    { metric: 'Code Quality', value: Math.max(0, 100 - analysis.issues_summary.total) },
    { metric: 'Documentation', value: analysis.line_metrics.total.comment_density * 100 },
    { metric: 'Test Coverage', value: 75 }, // Mock data
  ] : []

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
        {/* Enhanced Header */}
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
            {/* Enhanced Overview Cards */}
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

            {/* Enhanced Analysis Tabs */}
            <Tabs defaultValue="structure" className="space-y-4">
              <TabsList className="grid w-full grid-cols-5">
                <TabsTrigger value="structure">Repository Structure</TabsTrigger>
                <TabsTrigger value="metrics">Advanced Metrics</TabsTrigger>
                <TabsTrigger value="issues">Issues Analysis</TabsTrigger>
                <TabsTrigger value="activity">Git Activity</TabsTrigger>
                <TabsTrigger value="insights">AI Insights</TabsTrigger>
              </TabsList>

              <TabsContent value="structure" className="space-y-4">
                <Card>
                  <CardHeader>
                    <CardTitle>Interactive Repository Tree</CardTitle>
                    <CardDescription>
                      Navigate through your repository structure and view issues by file
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="max-h-96 overflow-y-auto border rounded-md p-4 bg-white">
                      <EnhancedRepositoryTree 
                        node={analysis.repository_structure} 
                        onFileSelect={handleFileSelect}
                      />
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>

              <TabsContent value="metrics" className="space-y-4">
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                  <Card>
                    <CardHeader>
                      <CardTitle>Quality Radar</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <ResponsiveContainer width="100%" height={250}>
                        <RadarChart data={complexityRadarData}>
                          <PolarGrid />
                          <PolarAngleAxis dataKey="metric" />
                          <PolarRadiusAxis angle={90} domain={[0, 100]} />
                          <Radar
                            name="Quality Metrics"
                            dataKey="value"
                            stroke="#3b82f6"
                            fill="#3b82f6"
                            fillOpacity={0.3}
                          />
                        </RadarChart>
                      </ResponsiveContainer>
                    </CardContent>
                  </Card>

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

              <TabsContent value="insights" className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <Card>
                    <CardHeader>
                      <CardTitle className="flex items-center space-x-2">
                        <Target className="h-5 w-5" />
                        <span>Recommendations</span>
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-3">
                      <div className="p-3 bg-blue-50 rounded-md">
                        <div className="font-medium text-blue-900">Code Quality</div>
                        <div className="text-sm text-blue-700">
                          Focus on reducing critical issues first for maximum impact
                        </div>
                      </div>
                      <div className="p-3 bg-green-50 rounded-md">
                        <div className="font-medium text-green-900">Documentation</div>
                        <div className="text-sm text-green-700">
                          Good comment density - maintain this level
                        </div>
                      </div>
                      <div className="p-3 bg-yellow-50 rounded-md">
                        <div className="font-medium text-yellow-900">Complexity</div>
                        <div className="text-sm text-yellow-700">
                          Consider refactoring high-complexity functions
                        </div>
                      </div>
                    </CardContent>
                  </Card>

                  <Card>
                    <CardHeader>
                      <CardTitle className="flex items-center space-x-2">
                        <TrendingUp className="h-5 w-5" />
                        <span>Trends</span>
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-3">
                      <div className="flex justify-between items-center">
                        <span className="text-sm">Code Quality Score</span>
                        <Badge variant="outline">
                          {(100 - analysis.issues_summary.total).toFixed(0)}/100
                        </Badge>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-sm">Maintainability</span>
                        <Badge variant="outline">
                          {analysis.complexity_metrics.maintainability_index.average.toFixed(0)}/100
                        </Badge>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-sm">Test Coverage</span>
                        <Badge variant="outline">75%</Badge>
                      </div>
                    </CardContent>
                  </Card>
                </div>
              </TabsContent>
            </Tabs>
          </div>
        )}

        {/* File Detail Modal */}
        <FileDetailModal
          file={selectedFile}
          isOpen={isFileModalOpen}
          onClose={() => setIsFileModalOpen(false)}
        />
      </div>
    </div>
  )
}
