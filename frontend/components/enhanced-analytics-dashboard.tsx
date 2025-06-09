"use client"

import React, { useState, useEffect } from 'react'
import { SessionProvider } from 'next-auth/react'
import { GitHubAuth } from './github-auth'
import { GitHubRepo } from '@/lib/github'
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
  BarChart3, 
  GitBranch, 
  RefreshCw, 
  XCircle, 
  CheckCircle, 
  Settings, 
  Zap, 
  Database,
  FileText,
  AlertTriangle,
  TrendingUp,
  Code,
  Users,
  Star,
  GitFork,
  Eye,
  ChevronDown,
  ChevronRight,
  Folder,
  FolderOpen,
  File,
  Activity,
  Calendar,
  Shield,
  Target,
  Layers,
  Cloud,
  Download,
  Share,
  Info
} from "lucide-react"
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  PieChart,
  Pie,
  Cell,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  LineChart,
  Line
} from 'recharts'

// Type definitions
interface AnalysisData {
  basic_metrics: {
    files: number
    functions: number
    classes: number
    modules: number
  }
  line_metrics: {
    total: {
      loc: number
      comment_density: number
    }
  }
  complexity_metrics: {
    cyclomatic_complexity: {
      average: number
    }
    maintainability_index: {
      average: number
    }
  }
  issues_summary: {
    total: number
    critical: number
    functional: number
    minor: number
  }
  issues: Array<{
    severity: string
    file: string
    description: string
    suggestion: string
  }>
  repository_structure: FileNode
  monthly_commits: Record<string, number>
}

// Utility functions
const getStatusColor = (status: string) => {
  switch (status) {
    case 'deployed': return 'text-green-600 bg-green-50'
    case 'deploying': return 'text-yellow-600 bg-yellow-50'
    case 'failed': return 'text-red-600 bg-red-50'
    default: return 'text-gray-600 bg-gray-50'
  }
}

interface FileNode {
  name: string
  type: 'file' | 'directory'
  children?: FileNode[]
  issues?: IssueItem[]
  path: string
  issue_count: number
  critical_issues: number
  functional_issues: number
  minor_issues: number
}

interface DeploymentConfig {
  mode: 'local' | 'modal' | 'docker'
  status: 'idle' | 'deploying' | 'deployed' | 'failed'
  endpoint: string
  lastDeployed: Date | null
}

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
          <div className="grid grid-cols-3 gap-4">
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
  onDeploy: (mode: 'local' | 'modal' | 'docker') => void;
}> = ({ config, onDeploy }) => {
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

// Main Enhanced Dashboard Component
export default function EnhancedAnalyticsDashboard() {
  const [repoUrl, setRepoUrl] = useState('')
  const [selectedRepo, setSelectedRepo] = useState<GitHubRepo | null>(null)
  const [analysisData, setAnalysisData] = useState<AnalysisData | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [selectedFile, setSelectedFile] = useState<FileNode | null>(null)
  const [isFileModalOpen, setIsFileModalOpen] = useState(false)
  const [deploymentConfig, setDeploymentConfig] = useState<DeploymentConfig>({
    mode: 'local',
    status: 'idle',
    endpoint: 'http://localhost:9997',
    lastDeployed: null
  })

  const analyzeRepository = async () => {
    if (!selectedRepo && !repoUrl.trim()) {
      setError('Please select a repository from GitHub or enter a repository URL')
      return
    }

    setLoading(true)
    setError('')
    
    try {
      const urlToAnalyze = selectedRepo ? selectedRepo.clone_url : repoUrl
      const response = await fetch('/api/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          repo_url: urlToAnalyze,
          repo_name: selectedRepo ? selectedRepo.name : urlToAnalyze.split('/').pop()?.replace('.git', '') || 'repository'
        }),
      })

      if (!response.ok) {
        throw new Error('Failed to analyze repository')
      }

      const data = await response.json()
      setAnalysisData(data)
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
        local: 'http://localhost:9997',
        modal: 'https://your-modal-app.modal.run',
        docker: 'http://localhost:9997'
      }
      
      setDeploymentConfig({
        mode,
        endpoint: endpoints[mode],
        status: 'deployed',
        lastDeployed: new Date()
      })
    }, 3000)
  }

  const handleFileSelect = (file: RepositoryNode) => {
    setSelectedFile(file)
    setIsFileModalOpen(true)
  }

  // Prepare enhanced chart data
  const complexityRadarData = analysisData ? [
    { metric: 'Complexity', value: analysisData.complexity_metrics.cyclomatic_complexity.average },
    { metric: 'Maintainability', value: analysisData.complexity_metrics.maintainability_index.average },
    { metric: 'Code Quality', value: Math.max(0, 100 - analysisData.issues_summary.total) },
    { metric: 'Documentation', value: analysisData.line_metrics.total.comment_density * 100 },
    { metric: 'Test Coverage', value: 75 }, // Mock data
  ] : []

  const issueDistributionData = analysisData ? [
    { name: 'Critical', value: analysisData.issues_summary.critical, color: '#ef4444' },
    { name: 'Functional', value: analysisData.issues_summary.functional, color: '#f59e0b' },
    { name: 'Minor', value: analysisData.issues_summary.minor, color: '#3b82f6' },
  ] : []

  const commitData = analysisData ? Object.entries(analysisData.monthly_commits).map(([month, commits]) => ({
    month,
    commits
  })) : []

  return (
    <SessionProvider>
      <div className="min-h-screen bg-gray-950 text-gray-100">
        <div className="container mx-auto p-6 space-y-6">
          <div className="text-center space-y-2">
            <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
              🔍 Enhanced Codebase Analytics
            </h1>
            <p className="text-gray-400 text-lg">
              Advanced repository analysis with deployment capabilities
            </p>
          </div>

          {/* GitHub Authentication */}
          <GitHubAuth 
            onRepoSelect={setSelectedRepo}
            selectedRepo={selectedRepo}
          />

          {/* Repository Input */}
          <Card className="bg-gray-900 border-gray-700">
            <CardHeader>
              <CardTitle className="text-gray-100">Repository Analysis</CardTitle>
              <CardDescription className="text-gray-400">
                {selectedRepo 
                  ? `Analyze ${selectedRepo.name} or enter a different repository URL`
                  : 'Enter a repository URL to analyze or select from your GitHub repositories above'
                }
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex gap-2">
                <Input
                  type="url"
                  placeholder="https://github.com/username/repository"
                  value={repoUrl}
                  onChange={(e) => setRepoUrl(e.target.value)}
                  className="flex-1 bg-gray-800 border-gray-600 text-gray-100 placeholder-gray-400"
                />
                <Button 
                  onClick={analyzeRepository} 
                  disabled={loading || (!selectedRepo && !repoUrl.trim())}
                  className="bg-blue-600 hover:bg-blue-700"
                >
                  {loading ? (
                    <>
                      <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                      Analyzing...
                    </>
                  ) : (
                    'Analyze Repository'
                  )}
                </Button>
              </div>
              {error && (
                <div className="text-red-400 text-sm bg-red-900/20 p-3 rounded border border-red-800">
                  {error}
                </div>
              )}
            </CardContent>
          </Card>

          {/* Deployment Status */}
          <DeploymentStatus 
            config={deploymentConfig}
            onDeploy={handleDeploy}
          />

          {/* Results */}
          {analysisData && (
            <div className="space-y-6">
              {/* Overview Cards */}
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                <Card className="bg-gradient-to-br from-blue-800 to-blue-900 border-blue-600">
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm font-medium text-blue-200">Total Files</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="text-3xl font-bold text-blue-100">{analysisData.basic_metrics.files}</div>
                    <div className="text-xs text-blue-300 mt-1">
                      {analysisData.basic_metrics.functions} functions, {analysisData.basic_metrics.classes} classes
                    </div>
                  </CardContent>
                </Card>

                <Card className="bg-gradient-to-br from-green-800 to-green-900 border-green-600">
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm font-medium text-green-200">Lines of Code</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="text-3xl font-bold text-green-100">{analysisData.line_metrics.total.loc.toLocaleString()}</div>
                    <div className="text-xs text-green-300 mt-1">
                      {(analysisData.line_metrics.total.comment_density * 100).toFixed(1)}% comments
                    </div>
                  </CardContent>
                </Card>

                <Card className="bg-gradient-to-br from-purple-800 to-purple-900 border-purple-600">
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm font-medium text-purple-200">Code Quality</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="text-3xl font-bold text-purple-100">
                      {analysisData.complexity_metrics.maintainability_index.average.toFixed(0)}
                    </div>
                    <div className="text-xs text-purple-300 mt-1">Maintainability Index</div>
                    <Progress 
                      value={analysisData.complexity_metrics.maintainability_index.average} 
                      className="mt-2"
                    />
                  </CardContent>
                </Card>

                <Card className="bg-gradient-to-br from-red-800 to-red-900 border-red-600">
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm font-medium text-red-200">Issues Found</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="text-3xl font-bold text-red-100">{analysisData.issues_summary.total}</div>
                    <div className="text-xs text-red-300 mt-1">
                      {analysisData.issues_summary.critical} critical, {analysisData.issues_summary.functional} functional
                    </div>
                  </CardContent>
                </Card>
              </div>

              {/* Analysis Tabs */}
              <Tabs defaultValue="structure" className="space-y-4">
                <TabsList className="grid w-full grid-cols-5 bg-gray-800">
                  <TabsTrigger value="structure" className="data-[state=active]:bg-gray-700 data-[state=active]:text-white">Structure</TabsTrigger>
                  <TabsTrigger value="issues" className="data-[state=active]:bg-gray-700 data-[state=active]:text-white">Issues</TabsTrigger>
                  <TabsTrigger value="complexity" className="data-[state=active]:bg-gray-700 data-[state=active]:text-white">Complexity</TabsTrigger>
                  <TabsTrigger value="metrics" className="data-[state=active]:bg-gray-700 data-[state=active]:text-white">Metrics</TabsTrigger>
                  <TabsTrigger value="deployment" className="data-[state=active]:bg-gray-700 data-[state=active]:text-white">Deploy</TabsTrigger>
                </TabsList>

                <TabsContent value="structure">
                  <Card className="bg-gray-800 border-gray-700">
                    <CardHeader>
                      <CardTitle className="text-white">Repository Structure</CardTitle>
                      <CardDescription className="text-gray-300">
                        Interactive file tree with detailed analysis
                      </CardDescription>
                    </CardHeader>
                    <CardContent>
                      <div className="max-h-96 overflow-y-auto border rounded-md p-4 bg-gray-900 border-gray-600">
                        <EnhancedRepositoryTree node={analysisData.repository_structure} level={0} />
                      </div>
                    </CardContent>
                  </Card>
                </TabsContent>

                <TabsContent value="issues">
                  <Card className="bg-gray-800 border-gray-700">
                    <CardHeader>
                      <CardTitle className="text-red-400">Issues Found</CardTitle>
                      <CardDescription className="text-gray-300">
                        Code quality issues and suggestions for improvement
                      </CardDescription>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-4">
                        {analysisData.issues.map((issue, index) => (
                          <div key={index} className="border-l-4 border-red-500 pl-4 py-2 bg-gray-900 rounded-r">
                            <div className="flex items-center gap-2 mb-1">
                              <Badge variant={issue.severity === 'critical' ? 'destructive' : 
                                           issue.severity === 'major' ? 'default' : 'secondary'}
                                     className={issue.severity === 'critical' ? 'bg-red-600' :
                                               issue.severity === 'major' ? 'bg-orange-600' : 'bg-gray-600'}>
                                {issue.severity}
                              </Badge>
                              <span className="text-sm text-gray-400">{issue.file}</span>
                            </div>
                            <p className="text-white font-medium">{issue.description}</p>
                            <p className="text-sm text-gray-400 mt-1">{issue.suggestion}</p>
                          </div>
                        ))}
                      </div>
                    </CardContent>
                  </Card>
                </TabsContent>

                <TabsContent value="complexity" className="space-y-4">
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

                <TabsContent value="metrics" className="space-y-4">
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                    <Card>
                      <CardHeader>
                        <CardTitle>Lines of Code</CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="text-3xl font-bold text-green-900">{analysisData.line_metrics.total.loc.toLocaleString()}</div>
                        <div className="text-xs text-green-600 mt-1">
                          {(analysisData.line_metrics.total.comment_density * 100).toFixed(1)}% comments
                        </div>
                      </CardContent>
                    </Card>

                    <Card>
                      <CardHeader>
                        <CardTitle>Complexity Metrics</CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="text-3xl font-bold text-purple-900">
                          {analysisData.complexity_metrics.cyclomatic_complexity.average.toFixed(0)}
                        </div>
                        <div className="text-xs text-purple-600 mt-1">Cyclomatic Complexity</div>
                        <Progress 
                          value={analysisData.complexity_metrics.cyclomatic_complexity.average} 
                          className="mt-2"
                        />
                      </CardContent>
                    </Card>
                  </div>
                </TabsContent>

                <TabsContent value="deployment" className="space-y-4">
                  <Card>
                    <CardHeader>
                      <CardTitle>Deployment Status</CardTitle>
                      <CardDescription>
                        Current deployment configuration and status
                      </CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <div className="flex items-center justify-between">
                        <div>
                          <div className="font-medium">Mode: {deploymentConfig.mode}</div>
                          <div className="text-sm text-gray-600">Endpoint: {deploymentConfig.endpoint}</div>
                        </div>
                        <Badge className={getStatusColor(deploymentConfig.status)}>
                          {deploymentConfig.status}
                        </Badge>
                      </div>
                      
                      <div className="grid grid-cols-3 gap-2">
                        <Button 
                          variant={deploymentConfig.mode === 'local' ? 'default' : 'outline'}
                          size="sm"
                          onClick={() => handleDeploy('local')}
                          disabled={deploymentConfig.status === 'deploying'}
                        >
                          <Settings size={16} className="mr-1" />
                          Local
                        </Button>
                        <Button 
                          variant={deploymentConfig.mode === 'modal' ? 'default' : 'outline'}
                          size="sm"
                          onClick={() => handleDeploy('modal')}
                          disabled={deploymentConfig.status === 'deploying'}
                        >
                          <Zap size={16} className="mr-1" />
                          Modal
                        </Button>
                        <Button 
                          variant={deploymentConfig.mode === 'docker' ? 'default' : 'outline'}
                          size="sm"
                          onClick={() => handleDeploy('docker')}
                          disabled={deploymentConfig.status === 'deploying'}
                        >
                          <Database size={16} className="mr-1" />
                          Docker
                        </Button>
                      </div>
                      
                      {deploymentConfig.status === 'deploying' && (
                        <div className="space-y-2">
                          <Progress value={66} className="w-full" />
                          <div className="text-sm text-gray-600">Deploying {deploymentConfig.mode} environment...</div>
                        </div>
                      )}
                    </CardContent>
                  </Card>
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
    </SessionProvider>
  )
}
