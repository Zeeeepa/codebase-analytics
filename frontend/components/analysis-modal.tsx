'use client'

import React, { useState, useMemo } from 'react'
import { 
  Dialog, 
  DialogContent, 
  DialogHeader, 
  DialogTitle, 
  DialogDescription 
} from '@/components/ui/dialog'
import { 
  Tabs, 
  TabsContent, 
  TabsList, 
  TabsTrigger 
} from '@/components/ui/tabs'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { 
  BarChart3, 
  FileText, 
  AlertTriangle, 
  CheckCircle, 
  XCircle, 
  Info,
  RefreshCw,
  FolderOpen,
  File,
  ChevronRight,
  ChevronDown
} from 'lucide-react'
import { AnalysisModalProps, RepositoryAnalysis, CodeIssue, FileNode } from '@/types/analysis'

// Repository Structure Tree Component
const RepositoryTree: React.FC<{ 
  node: FileNode; 
  level?: number;
  onFileSelect?: (file: FileNode) => void;
}> = ({ node, level = 0, onFileSelect }) => {
  const [isExpanded, setIsExpanded] = useState(level < 2)
  
  const totalIssues = node.issues.critical + node.issues.functional + node.issues.minor
  const hasIssues = totalIssues > 0
  
  const getIssueIcon = () => {
    if (node.issues.critical > 0) return <XCircle className="w-4 h-4 text-red-500" />
    if (node.issues.functional > 0) return <AlertTriangle className="w-4 h-4 text-yellow-500" />
    if (node.issues.minor > 0) return <Info className="w-4 h-4 text-blue-500" />
    return <CheckCircle className="w-4 h-4 text-green-500" />
  }

  return (
    <div className="space-y-1">
      <div 
        className={`flex items-center space-x-2 p-2 rounded hover:bg-gray-50 cursor-pointer ${
          hasIssues ? 'border-l-2 border-l-red-200' : ''
        }`}
        style={{ paddingLeft: `${level * 16 + 8}px` }}
        onClick={() => {
          if (node.type === 'directory') {
            setIsExpanded(!isExpanded)
          } else if (onFileSelect) {
            onFileSelect(node)
          }
        }}
      >
        {node.type === 'directory' && (
          isExpanded ? 
            <ChevronDown className="w-4 h-4 text-gray-400" /> : 
            <ChevronRight className="w-4 h-4 text-gray-400" />
        )}
        
        {node.type === 'directory' ? 
          <FolderOpen className="w-4 h-4 text-blue-500" /> : 
          <File className="w-4 h-4 text-gray-500" />
        }
        
        <span className="text-sm font-medium">{node.name}</span>
        
        {hasIssues && (
          <div className="flex items-center space-x-1 ml-auto">
            {getIssueIcon()}
            <span className="text-xs text-gray-600">
              {totalIssues} issue{totalIssues !== 1 ? 's' : ''}
            </span>
          </div>
        )}
        
        {hasIssues && (
          <div className="flex space-x-1">
            {node.issues.critical > 0 && (
              <Badge variant="destructive" className="text-xs px-1">
                {node.issues.critical}
              </Badge>
            )}
            {node.issues.functional > 0 && (
              <Badge variant="secondary" className="text-xs px-1 bg-yellow-100 text-yellow-800">
                {node.issues.functional}
              </Badge>
            )}
            {node.issues.minor > 0 && (
              <Badge variant="outline" className="text-xs px-1">
                {node.issues.minor}
              </Badge>
            )}
          </div>
        )}
      </div>
      
      {node.type === 'directory' && isExpanded && node.children && (
        <div>
          {node.children.map((child, index) => (
            <RepositoryTree 
              key={`${child.path}-${index}`}
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

// Metrics Dashboard Component
const MetricsDashboard: React.FC<{ analysis: RepositoryAnalysis }> = ({ analysis }) => {
  const { metrics, summary } = analysis
  
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-4 text-center">
            <div className="text-2xl font-bold text-blue-600">{metrics.files}</div>
            <div className="text-sm text-gray-600">Files</div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4 text-center">
            <div className="text-2xl font-bold text-green-600">{metrics.functions}</div>
            <div className="text-sm text-gray-600">Functions</div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4 text-center">
            <div className="text-2xl font-bold text-purple-600">{metrics.classes}</div>
            <div className="text-sm text-gray-600">Classes</div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4 text-center">
            <div className="text-2xl font-bold text-orange-600">{metrics.modules}</div>
            <div className="text-sm text-gray-600">Modules</div>
          </CardContent>
        </Card>
      </div>
      
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-4 text-center">
            <div className="text-2xl font-bold text-gray-800">{summary.total_issues}</div>
            <div className="text-sm text-gray-600">Total Issues</div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4 text-center">
            <div className="text-2xl font-bold text-red-600">{summary.critical_issues}</div>
            <div className="text-sm text-gray-600">Critical</div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4 text-center">
            <div className="text-2xl font-bold text-yellow-600">{summary.functional_issues}</div>
            <div className="text-sm text-gray-600">Functional</div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4 text-center">
            <div className="text-2xl font-bold text-blue-600">{summary.minor_issues}</div>
            <div className="text-sm text-gray-600">Minor</div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

// Issues List Component
const IssuesList: React.FC<{ 
  issues: CodeIssue[];
  selectedSeverity?: string;
}> = ({ issues, selectedSeverity }) => {
  const filteredIssues = useMemo(() => {
    if (!selectedSeverity || selectedSeverity === 'all') {
      return issues
    }
    return issues.filter(issue => issue.severity === selectedSeverity)
  }, [issues, selectedSeverity])

  const getSeverityIcon = (severity: string) => {
    switch (severity) {
      case 'critical': return <XCircle className="w-4 h-4 text-red-500" />
      case 'functional': return <AlertTriangle className="w-4 h-4 text-yellow-500" />
      case 'minor': return <Info className="w-4 h-4 text-blue-500" />
      default: return <Info className="w-4 h-4 text-gray-500" />
    }
  }

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical': return 'border-red-200 bg-red-50'
      case 'functional': return 'border-yellow-200 bg-yellow-50'
      case 'minor': return 'border-blue-200 bg-blue-50'
      default: return 'border-gray-200 bg-gray-50'
    }
  }

  return (
    <div className="space-y-3">
      {filteredIssues.map((issue, index) => (
        <Alert key={`${issue.id}-${index}`} className={getSeverityColor(issue.severity)}>
          <div className="flex items-start space-x-3">
            {getSeverityIcon(issue.severity)}
            <div className="flex-1 space-y-2">
              <div className="flex items-center justify-between">
                <h4 className="font-semibold text-sm">{issue.title}</h4>
                <Badge variant="outline" className="text-xs">
                  {issue.category}
                </Badge>
              </div>
              <AlertDescription className="text-xs">
                {issue.description}
              </AlertDescription>
              <div className="text-xs text-gray-600">
                📁 {issue.location.file}
                {issue.location.line && ` (Line ${issue.location.line})`}
                {issue.location.function && ` in ${issue.location.function}`}
              </div>
              {issue.code_snippet && (
                <pre className="text-xs bg-gray-100 p-2 rounded overflow-x-auto">
                  <code>{issue.code_snippet}</code>
                </pre>
              )}
              {issue.suggestion && (
                <div className="text-xs text-green-700 bg-green-50 p-2 rounded">
                  💡 <strong>Suggestion:</strong> {issue.suggestion}
                </div>
              )}
            </div>
          </div>
        </Alert>
      ))}
      {filteredIssues.length === 0 && (
        <div className="text-center py-8 text-gray-500">
          No issues found for the selected severity level.
        </div>
      )}
    </div>
  )
}

// Main Analysis Modal Component
export const AnalysisModal: React.FC<AnalysisModalProps> = ({ 
  analysis, 
  isOpen, 
  onClose, 
  onRefresh 
}) => {
  const [selectedSeverity, setSelectedSeverity] = useState<string>('all')
  const [selectedFile, setSelectedFile] = useState<FileNode | null>(null)

  if (!analysis) return null

  const handleFileSelect = (file: FileNode) => {
    setSelectedFile(file)
  }

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="max-w-6xl max-h-[90vh] overflow-hidden">
        <DialogHeader>
          <div className="flex items-center justify-between">
            <div>
              <DialogTitle className="flex items-center space-x-2">
                <BarChart3 size={24} />
                <span>📊 Repository Analysis Report</span>
              </DialogTitle>
              <DialogDescription>
                {analysis.repository.name} - Comprehensive code analysis and issue detection
              </DialogDescription>
            </div>
            {onRefresh && (
              <Button variant="outline" size="sm" onClick={onRefresh}>
                <RefreshCw className="w-4 h-4 mr-2" />
                Refresh
              </Button>
            )}
          </div>
        </DialogHeader>
        
        <Tabs defaultValue="overview" className="flex-1 overflow-hidden">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="structure">Structure</TabsTrigger>
            <TabsTrigger value="issues">Issues</TabsTrigger>
            <TabsTrigger value="details">Details</TabsTrigger>
          </TabsList>
          
          <div className="mt-4 h-[calc(90vh-200px)]">
            <TabsContent value="overview" className="h-full">
              <ScrollArea className="h-full">
                <div className="space-y-6">
                  <Card>
                    <CardHeader>
                      <CardTitle>Repository Metrics</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <MetricsDashboard analysis={analysis} />
                    </CardContent>
                  </Card>
                  
                  <Card>
                    <CardHeader>
                      <CardTitle>Issue Summary</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        <Alert className="border-red-200 bg-red-50">
                          <XCircle className="w-4 h-4 text-red-500" />
                          <AlertDescription>
                            <div className="font-semibold">Critical Issues</div>
                            <div className="text-sm">Implementation errors, misspelled function names, incorrect logic</div>
                          </AlertDescription>
                        </Alert>
                        <Alert className="border-yellow-200 bg-yellow-50">
                          <AlertTriangle className="w-4 h-4 text-yellow-500" />
                          <AlertDescription>
                            <div className="font-semibold">Functional Issues</div>
                            <div className="text-sm">Missing validation, incomplete implementations</div>
                          </AlertDescription>
                        </Alert>
                        <Alert className="border-blue-200 bg-blue-50">
                          <Info className="w-4 h-4 text-blue-500" />
                          <AlertDescription>
                            <div className="font-semibold">Minor Issues</div>
                            <div className="text-sm">Unused parameters, redundant code, formatting</div>
                          </AlertDescription>
                        </Alert>
                      </div>
                    </CardContent>
                  </Card>
                </div>
              </ScrollArea>
            </TabsContent>
            
            <TabsContent value="structure" className="h-full">
              <ScrollArea className="h-full">
                <Card>
                  <CardHeader>
                    <CardTitle>Repository Structure with Issue Count</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <RepositoryTree 
                      node={analysis.structure} 
                      onFileSelect={handleFileSelect}
                    />
                  </CardContent>
                </Card>
              </ScrollArea>
            </TabsContent>
            
            <TabsContent value="issues" className="h-full">
              <div className="space-y-4 h-full flex flex-col">
                <div className="flex items-center space-x-2">
                  <span className="text-sm font-medium">Filter by severity:</span>
                  <div className="flex space-x-2">
                    {['all', 'critical', 'functional', 'minor'].map((severity) => (
                      <Button
                        key={severity}
                        variant={selectedSeverity === severity ? 'default' : 'outline'}
                        size="sm"
                        onClick={() => setSelectedSeverity(severity)}
                      >
                        {severity.charAt(0).toUpperCase() + severity.slice(1)}
                      </Button>
                    ))}
                  </div>
                </div>
                <ScrollArea className="flex-1">
                  <IssuesList 
                    issues={analysis.issues} 
                    selectedSeverity={selectedSeverity}
                  />
                </ScrollArea>
              </div>
            </TabsContent>
            
            <TabsContent value="details" className="h-full">
              <ScrollArea className="h-full">
                <div className="space-y-4">
                  <Card>
                    <CardHeader>
                      <CardTitle>Analysis Details</CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <div>
                        <strong>Repository:</strong> {analysis.repository.name}
                      </div>
                      {analysis.repository.description && (
                        <div>
                          <strong>Description:</strong> {analysis.repository.description}
                        </div>
                      )}
                      <div>
                        <strong>Analysis Timestamp:</strong> {new Date(analysis.analysis_timestamp).toLocaleString()}
                      </div>
                      <div>
                        <strong>Total Files Analyzed:</strong> {analysis.metrics.files}
                      </div>
                      <div>
                        <strong>Code Elements Found:</strong>
                        <ul className="list-disc list-inside ml-4 mt-2">
                          <li>{analysis.metrics.functions} Functions</li>
                          <li>{analysis.metrics.classes} Classes</li>
                          <li>{analysis.metrics.modules} Modules</li>
                        </ul>
                      </div>
                    </CardContent>
                  </Card>
                  
                  {selectedFile && (
                    <Card>
                      <CardHeader>
                        <CardTitle>Selected File: {selectedFile.name}</CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="space-y-2">
                          <div><strong>Path:</strong> {selectedFile.path}</div>
                          <div><strong>Type:</strong> {selectedFile.type}</div>
                          {selectedFile.language && (
                            <div><strong>Language:</strong> {selectedFile.language}</div>
                          )}
                          {selectedFile.size && (
                            <div><strong>Size:</strong> {selectedFile.size} bytes</div>
                          )}
                          <div className="grid grid-cols-3 gap-2 mt-4">
                            <div className="text-center p-2 bg-red-50 rounded">
                              <div className="font-bold text-red-600">{selectedFile.issues.critical}</div>
                              <div className="text-xs">Critical</div>
                            </div>
                            <div className="text-center p-2 bg-yellow-50 rounded">
                              <div className="font-bold text-yellow-600">{selectedFile.issues.functional}</div>
                              <div className="text-xs">Functional</div>
                            </div>
                            <div className="text-center p-2 bg-blue-50 rounded">
                              <div className="font-bold text-blue-600">{selectedFile.issues.minor}</div>
                              <div className="text-xs">Minor</div>
                            </div>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  )}
                </div>
              </ScrollArea>
            </TabsContent>
          </div>
        </Tabs>
      </DialogContent>
    </Dialog>
  )
}

export default AnalysisModal

