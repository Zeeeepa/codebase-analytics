"use client"

import { VisualNode, Issue, IssueSeverity } from '@/lib/api-types'
import { 
  getSeverityColor, 
  getSeverityLabel, 
  getImpactBadgeInfo
} from '@/lib/analysis-utils'
import { useNodeSelection } from '@/hooks/useSharedAnalysisState'
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { FileCode, Code, FolderOpen, Package, ExternalLink, Eye, Target } from 'lucide-react'
import { useState } from 'react'

interface NodeCardProps {
  node: VisualNode
  showIssueCount?: boolean
  showBlastRadius?: boolean
  showMetadata?: boolean
  onViewIssues?: (node: VisualNode) => void
  onViewBlastRadius?: (node: VisualNode) => void
  onViewInExplorer?: (node: VisualNode) => void
  className?: string
}

export function NodeCard({
  node,
  showIssueCount = true,
  showBlastRadius = true,
  showMetadata = true,
  onViewIssues,
  onViewBlastRadius,
  onViewInExplorer,
  className = ''
}: NodeCardProps) {
  const { selectedNode, setSelectedNode } = useNodeSelection()
  
  const isSelected = selectedNode?.id === node.id
  
  const handleSelect = () => {
    setSelectedNode(isSelected ? null : node)
  }
  
  const getNodeIcon = (type: string) => {
    switch (type) {
      case 'file':
        return <FileCode className="h-5 w-5 text-blue-500" />;
      case 'function':
        return <Code className="h-5 w-5 text-purple-500" />;
      case 'class':
        return <Package className="h-5 w-5 text-orange-500" />;
      case 'module':
        return <FolderOpen className="h-5 w-5 text-green-500" />;
      default:
        return <FileCode className="h-5 w-5 text-blue-500" />;
    }
  }
  
  const renderIssueDistribution = () => {
    if (!node.issues || node.issues.length === 0) return null;
    
    // Count issues by severity
    const counts = node.issues.reduce((acc, issue) => {
      acc[issue.severity] = (acc[issue.severity] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);
    
    return (
      <div className="mt-2">
        <div className="text-xs text-muted-foreground mb-1">Issue severity</div>
        <div className="flex h-2 overflow-hidden rounded-full bg-muted">
          {Object.entries(counts).map(([severity, count]) => {
            const percentage = (count / node.issues.length) * 100;
            return percentage > 0 ? (
              <div 
                key={severity}
                className={`${getSeverityColor(severity)}`}
                style={{ width: `${percentage}%` }}
              />
            ) : null;
          })}
        </div>
      </div>
    );
  }
  
  return (
    <Card className={`overflow-hidden ${isSelected ? 'ring-2 ring-primary' : ''} ${className}`}>
      <CardHeader className="pb-2">
        <div className="flex items-center gap-2">
          {getNodeIcon(node.type)}
          <CardTitle className="text-lg">{node.name}</CardTitle>
          {showIssueCount && node.issues && node.issues.length > 0 && (
            <Badge variant="destructive">{node.issues.length} issues</Badge>
          )}
        </div>
        <CardDescription>{node.path}</CardDescription>
      </CardHeader>
      
      <CardContent className="pb-2">
        {showIssueCount && renderIssueDistribution()}
        
        {showBlastRadius && node.blast_radius > 0 && (
          <div className="mt-3 flex items-center gap-2">
            <Target className="h-4 w-4 text-orange-500" />
            <span className="text-sm">
              Blast Radius: <span className="font-medium">{node.blast_radius}</span> nodes
            </span>
          </div>
        )}
        
        {showMetadata && node.metadata && Object.keys(node.metadata).length > 0 && (
          <div className="mt-3 grid grid-cols-2 gap-2">
            {Object.entries(node.metadata).map(([key, value]) => (
              <div key={key} className="text-xs">
                <span className="text-muted-foreground">{key.replace(/_/g, ' ')}:</span>{' '}
                <span className="font-medium">{value}</span>
              </div>
            ))}
          </div>
        )}
      </CardContent>
      
      <CardFooter className="pt-2 flex items-center justify-between">
        <Button 
          variant="outline" 
          size="sm" 
          onClick={handleSelect}
        >
          {isSelected ? 'Deselect' : 'Select'}
        </Button>
        
        <div className="flex items-center gap-2">
          {onViewIssues && node.issues && node.issues.length > 0 && (
            <Button 
              variant="ghost" 
              size="sm"
              onClick={() => onViewIssues(node)}
            >
              <Eye className="h-4 w-4 mr-1" />
              Issues
            </Button>
          )}
          
          {onViewBlastRadius && (
            <Button 
              variant="ghost" 
              size="sm"
              onClick={() => onViewBlastRadius(node)}
            >
              <Target className="h-4 w-4 mr-1" />
              Blast Radius
            </Button>
          )}
          
          {onViewInExplorer && (
            <Button 
              variant="ghost" 
              size="sm"
              onClick={() => onViewInExplorer(node)}
            >
              <ExternalLink className="h-4 w-4 mr-1" />
              Explorer
            </Button>
          )}
        </div>
      </CardFooter>
    </Card>
  )
}

