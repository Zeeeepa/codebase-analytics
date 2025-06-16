"use client"

import { useState, useCallback, useMemo } from "react"
import { 
  ChevronRight, 
  ChevronDown, 
  File, 
  Folder, 
  FolderOpen,
  AlertTriangle,
  XCircle,
  AlertCircle,
  CheckCircle,
  Search,
  Filter,
  Eye,
  Code,
  Zap
} from "lucide-react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible"

interface StructuralError {
  file_path: string
  element_type: string
  element_name: string
  error_type: string
  reasoning: string
  severity: 'critical' | 'major' | 'minor'
  category: string
  line_number?: number
  suggestion?: string
  context?: string
}

interface StructuralNode {
  name: string
  type: 'repository' | 'directory' | 'file' | 'function' | 'class' | 'method'
  path: string
  children: StructuralNode[]
  errors: StructuralError[]
  error_count: {
    critical: number
    major: number
    minor: number
  }
  metrics: Record<string, any>
  metadata: Record<string, any>
}

interface AnalysisData {
  repository_info: {
    name: string
    total_files: number
    total_functions: number
    total_classes: number
    total_errors: number
  }
  error_summary: {
    by_severity: Record<string, number>
    by_category: Record<string, number>
    critical_count: number
    major_count: number
    minor_count: number
  }
  structural_tree: StructuralNode
  detailed_errors: Array<{
    index: number
    file_location: string
    error_place: string
    reasoning: string
    severity: string
    category: string
    line?: number
    suggestion?: string
  }>
  entry_points: Array<{
    name: string
    file: string
    type: string
    usage_count: number
  }>
  usage_heat_map: Record<string, number>
  inheritance_hierarchy: Record<string, string[]>
}

interface InteractiveStructuralTreeProps {
  analysisData: AnalysisData
  onNodeClick?: (node: StructuralNode) => void
  onErrorClick?: (error: StructuralError) => void
}

const SEVERITY_COLORS = {
  critical: '#ef4444',
  major: '#f59e0b', 
  minor: '#eab308',
  none: '#22c55e'
}

const SEVERITY_ICONS = {
  critical: XCircle,
  major: AlertTriangle,
  minor: AlertCircle,
  none: CheckCircle
}

export default function InteractiveStructuralTree({ 
  analysisData, 
  onNodeClick, 
  onErrorClick 
}: InteractiveStructuralTreeProps) {
  const [expandedNodes, setExpandedNodes] = useState<Set<string>>(new Set(['root']))
  const [selectedNode, setSelectedNode] = useState<StructuralNode | null>(null)
  const [searchTerm, setSearchTerm] = useState("")
  const [severityFilter, setSeverityFilter] = useState<string>("all")
  const [activeTab, setActiveTab] = useState("tree")

  const toggleNode = useCallback((path: string) => {
    setExpandedNodes(prev => {
      const newSet = new Set(prev)
      if (newSet.has(path)) {
        newSet.delete(path)
      } else {
        newSet.add(path)
      }
      return newSet
    })
  }, [])

  const handleNodeClick = useCallback((node: StructuralNode) => {
    setSelectedNode(node)
    onNodeClick?.(node)
  }, [onNodeClick])

  const handleErrorClick = useCallback((error: StructuralError) => {
    onErrorClick?.(error)
  }, [onErrorClick])

  const getNodeIcon = (node: StructuralNode, isExpanded: boolean) => {
    switch (node.type) {
      case 'repository':
      case 'directory':
        return isExpanded ? <FolderOpen className="h-4 w-4" /> : <Folder className="h-4 w-4" />
      case 'file':
        return <File className="h-4 w-4" />
      case 'function':
      case 'method':
        return <Code className="h-4 w-4" />
      case 'class':
        return <Zap className="h-4 w-4" />
      default:
        return <File className="h-4 w-4" />
    }
  }

  const getSeverityBadge = (errorCount: StructuralNode['error_count']) => {
    const total = errorCount.critical + errorCount.major + errorCount.minor
    if (total === 0) return null

    const severity = errorCount.critical > 0 ? 'critical' : 
                    errorCount.major > 0 ? 'major' : 'minor'
    
    const Icon = SEVERITY_ICONS[severity]
    
    return (
      <Badge 
        variant="outline" 
        className="ml-2 text-xs"
        style={{ 
          borderColor: SEVERITY_COLORS[severity], 
          color: SEVERITY_COLORS[severity] 
        }}
      >
        <Icon className="h-3 w-3 mr-1" />
        {total}
      </Badge>
    )
  }

  const filteredErrors = useMemo(() => {
    let errors = analysisData.detailed_errors
    
    if (searchTerm) {
      errors = errors.filter(error => 
        error.file_location.toLowerCase().includes(searchTerm.toLowerCase()) ||
        error.error_place.toLowerCase().includes(searchTerm.toLowerCase()) ||
        error.reasoning.toLowerCase().includes(searchTerm.toLowerCase())
      )
    }
    
    if (severityFilter !== "all") {
      errors = errors.filter(error => error.severity === severityFilter)
    }
    
    return errors
  }, [analysisData.detailed_errors, searchTerm, severityFilter])

  const renderTreeNode = (node: StructuralNode, level: number = 0) => {
    const isExpanded = expandedNodes.has(node.path)
    const hasChildren = node.children.length > 0
    const isSelected = selectedNode?.path === node.path

    return (
      <div key={node.path} className="select-none">
        <div 
          className={`flex items-center py-1 px-2 hover:bg-muted/50 cursor-pointer rounded-sm ${
            isSelected ? 'bg-primary/10 border-l-2 border-primary' : ''
          }`}
          style={{ paddingLeft: `${level * 20 + 8}px` }}
          onClick={() => handleNodeClick(node)}
        >
          {hasChildren ? (
            <Button
              variant="ghost"
              size="sm"
              className="h-4 w-4 p-0 mr-1"
              onClick={(e) => {
                e.stopPropagation()
                toggleNode(node.path)
              }}
            >
              {isExpanded ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
            </Button>
          ) : (
            <div className="w-5" />
          )}
          
          {getNodeIcon(node, isExpanded)}
          
          <span className="ml-2 text-sm font-medium">{node.name}</span>
          
          {getSeverityBadge(node.error_count)}
          
          {node.type === 'function' && node.metrics?.cyclomatic_complexity > 10 && (
            <Badge variant="outline" className="ml-2 text-xs border-orange-500 text-orange-500">
              Complex
            </Badge>
          )}
        </div>
        
        {hasChildren && isExpanded && (
          <div>
            {node.children.map(child => renderTreeNode(child, level + 1))}
          </div>
        )}
      </div>
    )
  }

  const renderErrorList = () => (
    <div className="space-y-2">
      {filteredErrors.map((error, index) => {
        const Icon = SEVERITY_ICONS[error.severity as keyof typeof SEVERITY_ICONS]
        return (
          <Card 
            key={index} 
            className="cursor-pointer hover:shadow-md transition-shadow"
            onClick={() => handleErrorClick(error as any)}
          >
            <CardContent className="p-4">
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="flex items-center mb-2">
                    <Icon 
                      className="h-4 w-4 mr-2" 
                      style={{ color: SEVERITY_COLORS[error.severity as keyof typeof SEVERITY_COLORS] }}
                    />
                    <Badge 
                      variant="outline"
                      style={{ 
                        borderColor: SEVERITY_COLORS[error.severity as keyof typeof SEVERITY_COLORS],
                        color: SEVERITY_COLORS[error.severity as keyof typeof SEVERITY_COLORS]
                      }}
                    >
                      {error.severity.toUpperCase()}
                    </Badge>
                    <span className="ml-2 text-xs text-muted-foreground">#{error.index}</span>
                  </div>
                  
                  <div className="space-y-1">
                    <p className="text-sm font-medium">{error.file_location}</p>
                    <p className="text-sm text-muted-foreground">{error.error_place}</p>
                    <p className="text-sm">{error.reasoning}</p>
                    
                    {error.suggestion && (
                      <div className="mt-2 p-2 bg-blue-50 rounded text-sm text-blue-800">
                        <strong>Suggestion:</strong> {error.suggestion}
                      </div>
                    )}
                  </div>
                </div>
                
                {error.line && (
                  <Badge variant="secondary" className="ml-2">
                    Line {error.line}
                  </Badge>
                )}
              </div>
            </CardContent>
          </Card>
        )
      })}
    </div>
  )

  const renderNodeDetails = () => {
    if (!selectedNode) {
      return (
        <div className="flex items-center justify-center h-64 text-muted-foreground">
          <div className="text-center">
            <Eye className="h-8 w-8 mx-auto mb-2" />
            <p>Select a node to view details</p>
          </div>
        </div>
      )
    }

    return (
      <div className="space-y-4">
        <div>
          <h3 className="text-lg font-semibold flex items-center">
            {getNodeIcon(selectedNode, false)}
            <span className="ml-2">{selectedNode.name}</span>
            {getSeverityBadge(selectedNode.error_count)}
          </h3>
          <p className="text-sm text-muted-foreground">{selectedNode.path}</p>
        </div>

        {Object.keys(selectedNode.metrics).length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle className="text-sm">Metrics</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              {Object.entries(selectedNode.metrics).map(([key, value]) => (
                <div key={key} className="flex justify-between text-sm">
                  <span className="capitalize">{key.replace(/_/g, ' ')}:</span>
                  <span className="font-medium">{String(value)}</span>
                </div>
              ))}
            </CardContent>
          </Card>
        )}

        {selectedNode.errors.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle className="text-sm">Errors ({selectedNode.errors.length})</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              {selectedNode.errors.map((error, index) => {
                const Icon = SEVERITY_ICONS[error.severity]
                return (
                  <div 
                    key={index} 
                    className="p-3 border rounded cursor-pointer hover:bg-muted/50"
                    onClick={() => handleErrorClick(error)}
                  >
                    <div className="flex items-center mb-1">
                      <Icon 
                        className="h-4 w-4 mr-2" 
                        style={{ color: SEVERITY_COLORS[error.severity] }}
                      />
                      <span className="text-sm font-medium">{error.error_type}</span>
                    </div>
                    <p className="text-sm text-muted-foreground">{error.reasoning}</p>
                    {error.suggestion && (
                      <p className="text-xs text-blue-600 mt-1">💡 {error.suggestion}</p>
                    )}
                  </div>
                )
              })}
            </CardContent>
          </Card>
        )}
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Summary Cards */}
      <div className="grid gap-4 md:grid-cols-4">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Total Files</p>
                <p className="text-2xl font-bold">{analysisData.repository_info.total_files}</p>
              </div>
              <File className="h-8 w-8 text-muted-foreground" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Functions</p>
                <p className="text-2xl font-bold">{analysisData.repository_info.total_functions}</p>
              </div>
              <Code className="h-8 w-8 text-muted-foreground" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Classes</p>
                <p className="text-2xl font-bold">{analysisData.repository_info.total_classes}</p>
              </div>
              <Zap className="h-8 w-8 text-muted-foreground" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Total Issues</p>
                <p className="text-2xl font-bold">{analysisData.repository_info.total_errors}</p>
              </div>
              <AlertTriangle className="h-8 w-8 text-muted-foreground" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Error Summary */}
      <Card>
        <CardHeader>
          <CardTitle>Error Summary</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 md:grid-cols-3">
            <div className="flex items-center space-x-2">
              <XCircle className="h-5 w-5 text-red-500" />
              <span className="text-sm">Critical: {analysisData.error_summary.critical_count}</span>
            </div>
            <div className="flex items-center space-x-2">
              <AlertTriangle className="h-5 w-5 text-orange-500" />
              <span className="text-sm">Major: {analysisData.error_summary.major_count}</span>
            </div>
            <div className="flex items-center space-x-2">
              <AlertCircle className="h-5 w-5 text-yellow-500" />
              <span className="text-sm">Minor: {analysisData.error_summary.minor_count}</span>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Main Content */}
      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="tree">Structural Tree</TabsTrigger>
          <TabsTrigger value="errors">Error List</TabsTrigger>
          <TabsTrigger value="analysis">Analysis</TabsTrigger>
        </TabsList>

        <TabsContent value="tree" className="space-y-4">
          <div className="grid gap-6 lg:grid-cols-3">
            {/* Tree View */}
            <div className="lg:col-span-2">
              <Card>
                <CardHeader>
                  <CardTitle>Project Structure</CardTitle>
                  <CardDescription>
                    Interactive codebase structure with error highlighting
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <ScrollArea className="h-96">
                    {renderTreeNode(analysisData.structural_tree)}
                  </ScrollArea>
                </CardContent>
              </Card>
            </div>

            {/* Details Panel */}
            <div>
              <Card>
                <CardHeader>
                  <CardTitle>Node Details</CardTitle>
                </CardHeader>
                <CardContent>
                  <ScrollArea className="h-96">
                    {renderNodeDetails()}
                  </ScrollArea>
                </CardContent>
              </Card>
            </div>
          </div>
        </TabsContent>

        <TabsContent value="errors" className="space-y-4">
          {/* Search and Filter */}
          <div className="flex gap-4">
            <div className="flex-1">
              <div className="relative">
                <Search className="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
                <Input
                  placeholder="Search errors..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="pl-8"
                />
              </div>
            </div>
            <select
              value={severityFilter}
              onChange={(e) => setSeverityFilter(e.target.value)}
              className="px-3 py-2 border rounded-md"
            >
              <option value="all">All Severities</option>
              <option value="critical">Critical</option>
              <option value="major">Major</option>
              <option value="minor">Minor</option>
            </select>
          </div>

          <ScrollArea className="h-96">
            {renderErrorList()}
          </ScrollArea>
        </TabsContent>

        <TabsContent value="analysis" className="space-y-4">
          <div className="grid gap-6 md:grid-cols-2">
            {/* Entry Points */}
            <Card>
              <CardHeader>
                <CardTitle>Entry Points</CardTitle>
                <CardDescription>Main functions and high-usage components</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  {analysisData.entry_points.map((entry, index) => (
                    <div key={index} className="flex justify-between items-center p-2 border rounded">
                      <div>
                        <p className="font-medium">{entry.name}</p>
                        <p className="text-sm text-muted-foreground">{entry.file}</p>
                      </div>
                      <Badge variant="secondary">
                        {entry.usage_count} uses
                      </Badge>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            {/* Usage Heat Map */}
            <Card>
              <CardHeader>
                <CardTitle>Usage Heat Map</CardTitle>
                <CardDescription>Most frequently used functions</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  {Object.entries(analysisData.usage_heat_map)
                    .sort(([,a], [,b]) => b - a)
                    .slice(0, 10)
                    .map(([name, count]) => (
                      <div key={name} className="flex justify-between items-center">
                        <span className="text-sm">{name}</span>
                        <div className="flex items-center space-x-2">
                          <div 
                            className="h-2 bg-blue-500 rounded"
                            style={{ width: `${Math.min(count * 10, 100)}px` }}
                          />
                          <span className="text-xs text-muted-foreground">{count}</span>
                        </div>
                      </div>
                    ))}
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  )
}

