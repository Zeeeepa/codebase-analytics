"use client"

import { useState, useCallback, useMemo, useEffect } from "react"
import { 
  Search, 
  AlertTriangle, 
  XCircle, 
  AlertCircle, 
  CheckCircle,
  Zap,
  Target,
  GitBranch,
  Network,
  Eye,
  Filter,
  Layers,
  MapPin,
  TrendingUp,
  Code,
  FileText,
  Settings
} from "lucide-react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"

interface VisualNode {
  id: string
  name: string
  type: 'function' | 'class' | 'file' | 'module'
  path: string
  issues: Array<{
    type: string
    severity: 'critical' | 'major' | 'minor'
    message: string
    suggestion?: string
  }>
  blast_radius: number
  impact_level: 'isolated' | 'local' | 'module' | 'system'
  metadata: Record<string, any>
  visual_properties: {
    color: string
    size: number
    border_width: number
    opacity: number
  }
}

interface VisualEdge {
  source: string
  target: string
  relationship: 'calls' | 'uses' | 'depends_on' | 'affects' | 'inherits'
  weight: number
  metadata: Record<string, any>
  visual_properties: {
    color: string
    width: number
    style: string
  }
}

interface ExplorationData {
  exploration_mode: string
  summary: {
    total_nodes: number
    total_issues: number
    error_hotspots_count: number
    critical_paths_count: number
    issue_distribution: Record<string, number>
  }
  visual_graph: {
    nodes: VisualNode[]
    edges: VisualEdge[]
  }
  error_hotspots: Array<{
    node_id: string
    name: string
    type: string
    issue_count: number
    blast_radius: number
    impact_level: string
  }>
  critical_paths: string[][]
  critical_nodes: Array<{
    id: string
    name: string
    type: string
    path: string
    issues: any[]
    blast_radius: number
    impact_level: string
  }>
  exploration_insights: Array<{
    type: string
    priority: string
    title: string
    description: string
    affected_nodes: string[]
  }>
}

interface VisualCodebaseExplorerProps {
  explorationData: ExplorationData
  onNodeClick?: (node: VisualNode) => void
  onModeChange?: (mode: string) => void
}

const EXPLORATION_MODES = [
  { value: "structural_overview", label: "Structural Overview", icon: Layers },
  { value: "error_focused", label: "Error Analysis", icon: AlertTriangle },
  { value: "blast_radius", label: "Impact Analysis", icon: Target },
  { value: "call_trace", label: "Call Flows", icon: GitBranch },
  { value: "dependency_map", label: "Dependencies", icon: Network },
  { value: "critical_paths", label: "Critical Paths", icon: MapPin }
]

const SEVERITY_COLORS = {
  critical: '#ef4444',
  major: '#f59e0b',
  minor: '#eab308'
}

const IMPACT_COLORS = {
  system: '#dc2626',
  module: '#ea580c',
  local: '#65a30d',
  isolated: '#22c55e'
}

export default function VisualCodebaseExplorer({ 
  explorationData, 
  onNodeClick, 
  onModeChange 
}: VisualCodebaseExplorerProps) {
  const [selectedMode, setSelectedMode] = useState(explorationData.exploration_mode)
  const [selectedNode, setSelectedNode] = useState<VisualNode | null>(null)
  const [searchTerm, setSearchTerm] = useState("")
  const [severityFilter, setSeverityFilter] = useState<string>("all")
  const [impactFilter, setImpactFilter] = useState<string>("all")
  const [activeView, setActiveView] = useState("overview")

  const handleModeChange = useCallback((mode: string) => {
    setSelectedMode(mode)
    onModeChange?.(mode)
  }, [onModeChange])

  const handleNodeClick = useCallback((node: VisualNode) => {
    setSelectedNode(node)
    onNodeClick?.(node)
  }, [onNodeClick])

  const filteredNodes = useMemo(() => {
    let nodes = explorationData.visual_graph.nodes

    if (searchTerm) {
      nodes = nodes.filter(node => 
        node.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        node.path.toLowerCase().includes(searchTerm.toLowerCase())
      )
    }

    if (severityFilter !== "all") {
      nodes = nodes.filter(node => 
        node.issues.some(issue => issue.severity === severityFilter)
      )
    }

    if (impactFilter !== "all") {
      nodes = nodes.filter(node => node.impact_level === impactFilter)
    }

    return nodes
  }, [explorationData.visual_graph.nodes, searchTerm, severityFilter, impactFilter])

  const getNodeIcon = (type: string) => {
    switch (type) {
      case 'function': return <Code className="h-4 w-4" />
      case 'class': return <Zap className="h-4 w-4" />
      case 'file': return <FileText className="h-4 w-4" />
      default: return <Settings className="h-4 w-4" />
    }
  }

  const getSeverityIcon = (severity: string) => {
    switch (severity) {
      case 'critical': return <XCircle className="h-4 w-4 text-red-500" />
      case 'major': return <AlertTriangle className="h-4 w-4 text-orange-500" />
      case 'minor': return <AlertCircle className="h-4 w-4 text-yellow-500" />
      default: return <CheckCircle className="h-4 w-4 text-green-500" />
    }
  }

  const renderModeSelector = () => (
    <div className="flex flex-wrap gap-2 mb-6">
      {EXPLORATION_MODES.map((mode) => {
        const Icon = mode.icon
        return (
          <Button
            key={mode.value}
            variant={selectedMode === mode.value ? "default" : "outline"}
            size="sm"
            onClick={() => handleModeChange(mode.value)}
            className="flex items-center gap-2"
          >
            <Icon className="h-4 w-4" />
            {mode.label}
          </Button>
        )
      })}
    </div>
  )

  const renderSummaryCards = () => (
    <div className="grid gap-4 md:grid-cols-4 mb-6">
      <Card>
        <CardContent className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-muted-foreground">Total Elements</p>
              <p className="text-2xl font-bold">{explorationData.summary.total_nodes}</p>
            </div>
            <Layers className="h-8 w-8 text-muted-foreground" />
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardContent className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-muted-foreground">Total Issues</p>
              <p className="text-2xl font-bold">{explorationData.summary.total_issues}</p>
            </div>
            <AlertTriangle className="h-8 w-8 text-muted-foreground" />
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardContent className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-muted-foreground">Error Hotspots</p>
              <p className="text-2xl font-bold">{explorationData.summary.error_hotspots_count}</p>
            </div>
            <Target className="h-8 w-8 text-muted-foreground" />
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardContent className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-muted-foreground">Critical Paths</p>
              <p className="text-2xl font-bold">{explorationData.summary.critical_paths_count}</p>
            </div>
            <MapPin className="h-8 w-8 text-muted-foreground" />
          </div>
        </CardContent>
      </Card>
    </div>
  )

  const renderFilters = () => (
    <div className="flex gap-4 mb-4">
      <div className="flex-1">
        <div className="relative">
          <Search className="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder="Search elements..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="pl-8"
          />
        </div>
      </div>
      
      <Select value={severityFilter} onValueChange={setSeverityFilter}>
        <SelectTrigger className="w-40">
          <SelectValue placeholder="Severity" />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="all">All Severities</SelectItem>
          <SelectItem value="critical">Critical</SelectItem>
          <SelectItem value="major">Major</SelectItem>
          <SelectItem value="minor">Minor</SelectItem>
        </SelectContent>
      </Select>

      <Select value={impactFilter} onValueChange={setImpactFilter}>
        <SelectTrigger className="w-40">
          <SelectValue placeholder="Impact" />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="all">All Impact</SelectItem>
          <SelectItem value="system">System</SelectItem>
          <SelectItem value="module">Module</SelectItem>
          <SelectItem value="local">Local</SelectItem>
          <SelectItem value="isolated">Isolated</SelectItem>
        </SelectContent>
      </Select>
    </div>
  )

  const renderErrorHotspots = () => (
    <div className="space-y-3">
      {explorationData.error_hotspots.slice(0, 10).map((hotspot, index) => (
        <Card key={hotspot.node_id} className="cursor-pointer hover:shadow-md transition-shadow">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-3">
                <div className="flex items-center space-x-2">
                  <span className="text-sm font-medium">#{index + 1}</span>
                  {getNodeIcon(hotspot.type)}
                  <span className="font-medium">{hotspot.name}</span>
                </div>
                <Badge 
                  variant="outline"
                  style={{ 
                    borderColor: IMPACT_COLORS[hotspot.impact_level as keyof typeof IMPACT_COLORS],
                    color: IMPACT_COLORS[hotspot.impact_level as keyof typeof IMPACT_COLORS]
                  }}
                >
                  {hotspot.impact_level}
                </Badge>
              </div>
              
              <div className="flex items-center space-x-4 text-sm text-muted-foreground">
                <span>{hotspot.issue_count} issues</span>
                <span>Blast radius: {hotspot.blast_radius}</span>
              </div>
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  )

  const renderCriticalNodes = () => (
    <div className="space-y-3">
      {explorationData.critical_nodes.slice(0, 10).map((node, index) => (
        <Card 
          key={node.id} 
          className="cursor-pointer hover:shadow-md transition-shadow"
          onClick={() => {
            const fullNode = explorationData.visual_graph.nodes.find(n => n.id === node.id)
            if (fullNode) handleNodeClick(fullNode)
          }}
        >
          <CardContent className="p-4">
            <div className="flex items-start justify-between">
              <div className="flex-1">
                <div className="flex items-center space-x-2 mb-2">
                  {getNodeIcon(node.type)}
                  <span className="font-medium">{node.name}</span>
                  <Badge variant="secondary">{node.type}</Badge>
                </div>
                
                <p className="text-sm text-muted-foreground mb-2">{node.path}</p>
                
                {node.issues.length > 0 && (
                  <div className="flex flex-wrap gap-1">
                    {node.issues.slice(0, 3).map((issue, idx) => (
                      <div key={idx} className="flex items-center space-x-1">
                        {getSeverityIcon(issue.severity)}
                        <span className="text-xs">{issue.type}</span>
                      </div>
                    ))}
                    {node.issues.length > 3 && (
                      <span className="text-xs text-muted-foreground">
                        +{node.issues.length - 3} more
                      </span>
                    )}
                  </div>
                )}
              </div>
              
              <div className="text-right">
                <div className="text-sm font-medium">Impact: {node.blast_radius}</div>
                <Badge 
                  variant="outline"
                  style={{ 
                    borderColor: IMPACT_COLORS[node.impact_level as keyof typeof IMPACT_COLORS],
                    color: IMPACT_COLORS[node.impact_level as keyof typeof IMPACT_COLORS]
                  }}
                >
                  {node.impact_level}
                </Badge>
              </div>
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  )

  const renderInsights = () => (
    <div className="space-y-4">
      {explorationData.exploration_insights.map((insight, index) => (
        <Card key={index}>
          <CardContent className="p-4">
            <div className="flex items-start space-x-3">
              <div className="flex-shrink-0">
                {insight.priority === 'critical' && <XCircle className="h-5 w-5 text-red-500" />}
                {insight.priority === 'medium' && <AlertTriangle className="h-5 w-5 text-orange-500" />}
                {insight.priority === 'low' && <AlertCircle className="h-5 w-5 text-yellow-500" />}
              </div>
              
              <div className="flex-1">
                <div className="flex items-center space-x-2 mb-1">
                  <h4 className="font-medium">{insight.title}</h4>
                  <Badge 
                    variant={insight.priority === 'critical' ? 'destructive' : 'secondary'}
                  >
                    {insight.priority}
                  </Badge>
                </div>
                
                <p className="text-sm text-muted-foreground mb-2">{insight.description}</p>
                
                <div className="text-xs text-muted-foreground">
                  Affects {insight.affected_nodes.length} elements
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      ))}
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
          <div className="flex items-center space-x-2 mb-2">
            {getNodeIcon(selectedNode.type)}
            <h3 className="text-lg font-semibold">{selectedNode.name}</h3>
            <Badge variant="secondary">{selectedNode.type}</Badge>
          </div>
          <p className="text-sm text-muted-foreground">{selectedNode.path}</p>
        </div>

        {selectedNode.metadata && Object.keys(selectedNode.metadata).length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle className="text-sm">Metrics</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              {Object.entries(selectedNode.metadata).map(([key, value]) => (
                <div key={key} className="flex justify-between text-sm">
                  <span className="capitalize">{key.replace(/_/g, ' ')}:</span>
                  <span className="font-medium">{String(value)}</span>
                </div>
              ))}
              <div className="flex justify-between text-sm">
                <span>Blast Radius:</span>
                <span className="font-medium">{selectedNode.blast_radius}</span>
              </div>
              <div className="flex justify-between text-sm">
                <span>Impact Level:</span>
                <Badge 
                  variant="outline"
                  style={{ 
                    borderColor: IMPACT_COLORS[selectedNode.impact_level as keyof typeof IMPACT_COLORS],
                    color: IMPACT_COLORS[selectedNode.impact_level as keyof typeof IMPACT_COLORS]
                  }}
                >
                  {selectedNode.impact_level}
                </Badge>
              </div>
            </CardContent>
          </Card>
        )}

        {selectedNode.issues.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle className="text-sm">Issues ({selectedNode.issues.length})</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              {selectedNode.issues.map((issue, index) => (
                <div key={index} className="p-3 border rounded">
                  <div className="flex items-center space-x-2 mb-1">
                    {getSeverityIcon(issue.severity)}
                    <span className="text-sm font-medium">{issue.type}</span>
                    <Badge 
                      variant="outline"
                      style={{ 
                        borderColor: SEVERITY_COLORS[issue.severity as keyof typeof SEVERITY_COLORS],
                        color: SEVERITY_COLORS[issue.severity as keyof typeof SEVERITY_COLORS]
                      }}
                    >
                      {issue.severity}
                    </Badge>
                  </div>
                  <p className="text-sm text-muted-foreground mb-1">{issue.message}</p>
                  {issue.suggestion && (
                    <p className="text-xs text-blue-600">💡 {issue.suggestion}</p>
                  )}
                </div>
              ))}
            </CardContent>
          </Card>
        )}
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Mode Selector */}
      {renderModeSelector()}

      {/* Summary Cards */}
      {renderSummaryCards()}

      {/* Main Content */}
      <Tabs value={activeView} onValueChange={setActiveView}>
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="hotspots">Error Hotspots</TabsTrigger>
          <TabsTrigger value="critical">Critical Nodes</TabsTrigger>
          <TabsTrigger value="insights">Insights</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-4">
          {renderFilters()}
          
          <div className="grid gap-6 lg:grid-cols-3">
            <div className="lg:col-span-2">
              <Card>
                <CardHeader>
                  <CardTitle>Visual Graph</CardTitle>
                  <CardDescription>
                    Interactive visualization of codebase structure and relationships
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="h-96 bg-muted/20 rounded-lg flex items-center justify-center">
                    <div className="text-center text-muted-foreground">
                      <Network className="h-12 w-12 mx-auto mb-2" />
                      <p>Interactive graph visualization</p>
                      <p className="text-sm">Showing {filteredNodes.length} nodes</p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>

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

        <TabsContent value="hotspots" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Error Hotspots</CardTitle>
              <CardDescription>
                Areas with the highest concentration of issues and impact
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ScrollArea className="h-96">
                {renderErrorHotspots()}
              </ScrollArea>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="critical" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Critical Nodes</CardTitle>
              <CardDescription>
                High-impact elements that require immediate attention
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ScrollArea className="h-96">
                {renderCriticalNodes()}
              </ScrollArea>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="insights" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Exploration Insights</CardTitle>
              <CardDescription>
                Actionable recommendations based on codebase analysis
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ScrollArea className="h-96">
                {renderInsights()}
              </ScrollArea>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}

