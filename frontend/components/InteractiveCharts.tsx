"use client"

import { useState, useCallback } from "react"
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  LineChart,
  Line,
  ScatterChart,
  Scatter,
  Legend
} from "recharts"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { 
  ZoomIn, 
  ZoomOut, 
  Filter, 
  TrendingUp, 
  BarChart3, 
  PieChart as PieChartIcon,
  Activity,
  AlertTriangle,
  CheckCircle,
  XCircle
} from "lucide-react"

interface InteractiveChartsProps {
  commitData: Array<{ month: string; commits: number }>
  repoData: {
    maintainabilityIndex: number
    cyclomaticComplexity: number
    halsteadVolume: number
    commentDensity: number
    numberOfFiles: number
    numberOfFunctions: number
    numberOfClasses: number
    linesOfCode: number
  }
  onDrillDown?: (metric: string, value: any) => void
}

interface MetricData {
  name: string
  value: number
  color: string
  severity: 'good' | 'warning' | 'critical'
  description: string
}

const COLORS = {
  good: '#22c55e',
  warning: '#f59e0b', 
  critical: '#ef4444',
  primary: '#2563eb',
  secondary: '#64748b'
}

export default function InteractiveCharts({ 
  commitData, 
  repoData, 
  onDrillDown 
}: InteractiveChartsProps) {
  const [selectedMetric, setSelectedMetric] = useState<string | null>(null)
  const [zoomLevel, setZoomLevel] = useState(1)
  const [activeTab, setActiveTab] = useState("overview")

  // Transform repo data into interactive metrics
  const metricsData: MetricData[] = [
    {
      name: "Maintainability",
      value: repoData.maintainabilityIndex,
      color: repoData.maintainabilityIndex >= 70 ? COLORS.good : 
             repoData.maintainabilityIndex >= 50 ? COLORS.warning : COLORS.critical,
      severity: repoData.maintainabilityIndex >= 70 ? 'good' : 
                repoData.maintainabilityIndex >= 50 ? 'warning' : 'critical',
      description: "Code maintainability index (0-100)"
    },
    {
      name: "Complexity",
      value: repoData.cyclomaticComplexity,
      color: repoData.cyclomaticComplexity <= 10 ? COLORS.good :
             repoData.cyclomaticComplexity <= 20 ? COLORS.warning : COLORS.critical,
      severity: repoData.cyclomaticComplexity <= 10 ? 'good' :
                repoData.cyclomaticComplexity <= 20 ? 'warning' : 'critical',
      description: "Average cyclomatic complexity"
    },
    {
      name: "Documentation",
      value: repoData.commentDensity,
      color: repoData.commentDensity >= 20 ? COLORS.good :
             repoData.commentDensity >= 10 ? COLORS.warning : COLORS.critical,
      severity: repoData.commentDensity >= 20 ? 'good' :
                repoData.commentDensity >= 10 ? 'warning' : 'critical',
      description: "Comment density percentage"
    }
  ]

  // Code distribution data
  const codeDistribution = [
    { name: "Functions", value: repoData.numberOfFunctions, color: COLORS.primary },
    { name: "Classes", value: repoData.numberOfClasses, color: COLORS.secondary },
    { name: "Files", value: repoData.numberOfFiles, color: COLORS.good }
  ]

  // Complexity vs Size scatter data
  const complexityData = [
    {
      complexity: repoData.cyclomaticComplexity,
      size: repoData.linesOfCode / 1000, // Convert to thousands
      maintainability: repoData.maintainabilityIndex,
      name: "Current Codebase"
    }
  ]

  const handleMetricClick = useCallback((metric: string, value: any) => {
    setSelectedMetric(metric)
    onDrillDown?.(metric, value)
  }, [onDrillDown])

  const handleZoomIn = () => setZoomLevel(prev => Math.min(prev * 1.2, 3))
  const handleZoomOut = () => setZoomLevel(prev => Math.max(prev / 1.2, 0.5))

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-background border rounded-lg p-3 shadow-lg">
          <p className="font-medium">{label}</p>
          {payload.map((entry: any, index: number) => (
            <p key={index} style={{ color: entry.color }}>
              {entry.name}: {entry.value}
            </p>
          ))}
          <p className="text-xs text-muted-foreground mt-1">
            Click to explore details
          </p>
        </div>
      )
    }
    return null
  }

  const MetricCard = ({ metric }: { metric: MetricData }) => {
    const Icon = metric.severity === 'good' ? CheckCircle :
                metric.severity === 'warning' ? AlertTriangle : XCircle

    return (
      <Card 
        className="cursor-pointer hover:shadow-md transition-shadow"
        onClick={() => handleMetricClick(metric.name, metric.value)}
      >
        <CardContent className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-muted-foreground">{metric.name}</p>
              <p className="text-2xl font-bold">{metric.value.toFixed(1)}</p>
              <p className="text-xs text-muted-foreground">{metric.description}</p>
            </div>
            <div className="flex flex-col items-center">
              <Icon 
                className="h-6 w-6 mb-1" 
                style={{ color: metric.color }}
              />
              <Badge 
                variant={metric.severity === 'good' ? 'default' : 'destructive'}
                className="text-xs"
              >
                {metric.severity}
              </Badge>
            </div>
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <div className="space-y-6">
      {/* Interactive Controls */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <Button
            variant="outline"
            size="sm"
            onClick={handleZoomOut}
            disabled={zoomLevel <= 0.5}
          >
            <ZoomOut className="h-4 w-4" />
          </Button>
          <span className="text-sm text-muted-foreground">
            Zoom: {Math.round(zoomLevel * 100)}%
          </span>
          <Button
            variant="outline"
            size="sm"
            onClick={handleZoomIn}
            disabled={zoomLevel >= 3}
          >
            <ZoomIn className="h-4 w-4" />
          </Button>
        </div>
        <div className="flex items-center space-x-2">
          <Filter className="h-4 w-4 text-muted-foreground" />
          <span className="text-sm text-muted-foreground">
            {selectedMetric ? `Focused on: ${selectedMetric}` : 'All metrics'}
          </span>
        </div>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="trends">Trends</TabsTrigger>
          <TabsTrigger value="distribution">Distribution</TabsTrigger>
          <TabsTrigger value="analysis">Analysis</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-4">
          {/* Interactive Metrics Cards */}
          <div className="grid gap-4 md:grid-cols-3">
            {metricsData.map((metric) => (
              <MetricCard key={metric.name} metric={metric} />
            ))}
          </div>

          {/* Enhanced Commit Chart */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center">
                <TrendingUp className="h-5 w-5 mr-2" />
                Interactive Commit Activity
              </CardTitle>
              <CardDescription>
                Click on bars to explore commit details for specific months
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300 * zoomLevel}>
                <BarChart data={commitData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="month" />
                  <YAxis />
                  <Tooltip content={<CustomTooltip />} />
                  <Bar 
                    dataKey="commits" 
                    fill={COLORS.primary}
                    onClick={(data) => handleMetricClick('commits', data)}
                    className="cursor-pointer hover:opacity-80"
                  />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="trends" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Quality Trends</CardTitle>
              <CardDescription>
                Visualize how different metrics relate to each other
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={metricsData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" />
                  <YAxis />
                  <Tooltip />
                  <Line 
                    type="monotone" 
                    dataKey="value" 
                    stroke={COLORS.primary}
                    strokeWidth={2}
                    dot={{ fill: COLORS.primary, strokeWidth: 2, r: 6 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="distribution" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center">
                <PieChartIcon className="h-5 w-5 mr-2" />
                Code Distribution
              </CardTitle>
              <CardDescription>
                Interactive breakdown of codebase components
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={codeDistribution}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="value"
                    onClick={(data) => handleMetricClick('distribution', data)}
                    className="cursor-pointer"
                  >
                    {codeDistribution.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="analysis" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center">
                <Activity className="h-5 w-5 mr-2" />
                Complexity vs Maintainability
              </CardTitle>
              <CardDescription>
                Scatter plot showing the relationship between complexity and maintainability
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <ScatterChart data={complexityData}>
                  <CartesianGrid />
                  <XAxis 
                    type="number" 
                    dataKey="complexity" 
                    name="Complexity"
                    label={{ value: 'Cyclomatic Complexity', position: 'insideBottom', offset: -10 }}
                  />
                  <YAxis 
                    type="number" 
                    dataKey="maintainability" 
                    name="Maintainability"
                    label={{ value: 'Maintainability Index', angle: -90, position: 'insideLeft' }}
                  />
                  <Tooltip 
                    cursor={{ strokeDasharray: '3 3' }}
                    content={({ active, payload }) => {
                      if (active && payload && payload.length) {
                        const data = payload[0].payload
                        return (
                          <div className="bg-background border rounded-lg p-3 shadow-lg">
                            <p className="font-medium">{data.name}</p>
                            <p>Complexity: {data.complexity.toFixed(1)}</p>
                            <p>Maintainability: {data.maintainability.toFixed(1)}</p>
                            <p>Size: {(data.size * 1000).toLocaleString()} LOC</p>
                          </div>
                        )
                      }
                      return null
                    }}
                  />
                  <Scatter 
                    name="Codebase" 
                    dataKey="maintainability" 
                    fill={COLORS.primary}
                    onClick={(data) => handleMetricClick('complexity-analysis', data)}
                    className="cursor-pointer"
                  />
                </ScatterChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* Selected Metric Details */}
      {selectedMetric && (
        <Card className="border-primary">
          <CardHeader>
            <CardTitle className="flex items-center justify-between">
              <span>Exploring: {selectedMetric}</span>
              <Button 
                variant="ghost" 
                size="sm"
                onClick={() => setSelectedMetric(null)}
              >
                ✕
              </Button>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-muted-foreground">
              Click on any chart element to drill down into specific details. 
              This interactive exploration helps you understand your codebase metrics better.
            </p>
          </CardContent>
        </Card>
      )}
    </div>
  )
}

