"use client"

import { useState, useEffect } from "react"
import { Search, Code2, AlertTriangle, Target, Zap, FileCode, GitBranch, Bug, Settings } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Progress } from "@/components/ui/progress"

interface VisualNode {
  id: string
  name: string
  type: string
  path: string
  issues: Issue[]
  blast_radius: number
  metadata: any
}

interface Issue {
  type: string
  severity: string
  message: string
  suggestion: string
}

interface ExplorationData {
  summary: {
    total_nodes: number
    total_issues: number
    error_hotspots_count: number
    critical_paths_count: number
  }
  error_hotspots: VisualNode[]
  exploration_insights: Insight[]
  critical_paths: any[]
}

interface Insight {
  type: string
  priority: string
  title: string
  description: string
  affected_nodes: string[]
}

interface BlastRadiusData {
  target_symbol: VisualNode
  blast_radius: {
    affected_nodes: number
    affected_edges: number
  }
}

export default function InteractiveCodebaseExplorer() {
  const [repoUrl, setRepoUrl] = useState("")
  const [analysisMode, setAnalysisMode] = useState("error_focused")
  const [isLoading, setIsLoading] = useState(false)
  const [explorationData, setExplorationData] = useState<ExplorationData | null>(null)
  const [blastRadiusData, setBlastRadiusData] = useState<BlastRadiusData | null>(null)
  const [selectedSymbol, setSelectedSymbol] = useState("")
  const [error, setError] = useState("")

  const getBackendUrl = () => {
    if (typeof window !== 'undefined' && window.location.hostname === 'localhost') {
      return 'http://localhost:8000'
    }
    return 'https://zeeeepa--analytics-app-fastapi-modal-app-dev.modal.run'
  }

  const handleVisualExploration = async () => {
    if (!repoUrl.trim()) {
      setError("Please enter a repository URL or use '.' for current directory")
      return
    }

    setIsLoading(true)
    setError("")
    
    try {
      const response = await fetch(`${getBackendUrl()}/explore_visual`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          repo_url: repoUrl,
          mode: analysisMode
        })
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()
      setExplorationData(data)
    } catch (error) {
      console.error('Error during visual exploration:', error)
      setError('Failed to analyze repository. Please check the URL and try again.')
    } finally {
      setIsLoading(false)
    }
  }

  const handleBlastRadiusAnalysis = async () => {
    if (!selectedSymbol.trim()) {
      setError("Please enter a symbol name for blast radius analysis")
      return
    }

    setIsLoading(true)
    setError("")
    
    try {
      const response = await fetch(`${getBackendUrl()}/analyze_blast_radius`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          repo_url: repoUrl || '.',
          symbol_name: selectedSymbol
        })
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()
      setBlastRadiusData(data)
    } catch (error) {
      console.error('Error during blast radius analysis:', error)
      setError('Failed to analyze symbol impact. Please check the symbol name and try again.')
    } finally {
      setIsLoading(false)
    }
  }

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical': return 'bg-red-500'
      case 'major': return 'bg-orange-500'
      case 'minor': return 'bg-yellow-500'
      default: return 'bg-gray-500'
    }
  }

  const getIssueTypeIcon = (type: string) => {
    switch (type) {
      case 'unused_parameter':
      case 'unused_function':
        return <Code2 className="h-4 w-4" />
      case 'mutable_default_parameter':
      case 'missing_required_arguments':
        return <AlertTriangle className="h-4 w-4" />
      case 'undefined_function_call':
        return <Bug className="h-4 w-4" />
      default:
        return <FileCode className="h-4 w-4" />
    }
  }

  return (
    <div className="container mx-auto p-6 space-y-6">
      <div className="text-center space-y-2">
        <h1 className="text-3xl font-bold">Interactive Codebase Explorer</h1>
        <p className="text-muted-foreground">
          Analyze codebases for functional errors, parameter issues, and architectural problems
        </p>
      </div>

      {/* Input Controls */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Search className="h-5 w-5" />
            Repository Analysis
          </CardTitle>
          <CardDescription>
            Enter a repository URL or use '.' for the current directory
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex gap-2">
            <Input
              placeholder="Repository URL (e.g., https://github.com/user/repo or '.')"
              value={repoUrl}
              onChange={(e) => setRepoUrl(e.target.value)}
              className="flex-1"
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
          </div>
          
          <div className="flex gap-2">
            <Button 
              onClick={handleVisualExploration} 
              disabled={isLoading}
              className="flex items-center gap-2"
            >
              <Zap className="h-4 w-4" />
              {isLoading ? 'Analyzing...' : 'Start Visual Analysis'}
            </Button>
            
            <div className="flex gap-2 flex-1">
              <Input
                placeholder="Symbol name for blast radius analysis"
                value={selectedSymbol}
                onChange={(e) => setSelectedSymbol(e.target.value)}
                className="flex-1"
              />
              <Button 
                onClick={handleBlastRadiusAnalysis} 
                disabled={isLoading}
                variant="outline"
                className="flex items-center gap-2"
              >
                <Target className="h-4 w-4" />
                Blast Radius
              </Button>
            </div>
          </div>

          {error && (
            <Alert variant="destructive">
              <AlertTriangle className="h-4 w-4" />
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}
        </CardContent>
      </Card>

      {/* Loading Progress */}
      {isLoading && (
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
      )}

      {/* Results Tabs */}
      {(explorationData || blastRadiusData) && (
        <Tabs defaultValue="exploration" className="space-y-4">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="exploration" disabled={!explorationData}>
              Visual Exploration
            </TabsTrigger>
            <TabsTrigger value="blast-radius" disabled={!blastRadiusData}>
              Blast Radius Analysis
            </TabsTrigger>
          </TabsList>

          {/* Visual Exploration Results */}
          <TabsContent value="exploration" className="space-y-4">
            {explorationData && (
              <>
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
                      <Settings className="h-5 w-5 text-blue-500" />
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
              </>
            )}
          </TabsContent>

          {/* Blast Radius Results */}
          <TabsContent value="blast-radius" className="space-y-4">
            {blastRadiusData && (
              <>
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
                  </CardContent>
                </Card>
              </>
            )}
          </TabsContent>
        </Tabs>
      )}
    </div>
  )
}

