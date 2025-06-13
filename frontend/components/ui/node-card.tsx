"use client"

import { useState } from 'react';
import { VisualNode, IssueSeverity } from '@/lib/api-types';
import { useNodeSelection } from '@/hooks/useAnalysisState';
import { 
  Code2, 
  FileCode, 
  Target, 
  ExternalLink, 
  ChevronDown, 
  ChevronUp,
  AlertTriangle,
  Box,
  Boxes,
  Component,
  Database,
  File,
  FolderTree
} from 'lucide-react';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible';
import { cn } from '@/lib/utils';

interface NodeCardProps {
  node: VisualNode;
  showIssueCount?: boolean;
  showBlastRadius?: boolean;
  showMetadata?: boolean;
  onViewIssues?: (node: VisualNode) => void;
  onViewBlastRadius?: (node: VisualNode) => void;
  onViewInExplorer?: (node: VisualNode) => void;
  className?: string;
}

export function NodeCard({
  node,
  showIssueCount = true,
  showBlastRadius = true,
  showMetadata = true,
  onViewIssues,
  onViewBlastRadius,
  onViewInExplorer,
  className
}: NodeCardProps) {
  const [isOpen, setIsOpen] = useState(false);
  const { selectedNode, setSelectedNode } = useNodeSelection();
  
  const isSelected = selectedNode?.id === node.id;
  
  const handleSelect = () => {
    setSelectedNode(isSelected ? null : node);
  };
  
  const getNodeTypeIcon = (type: string) => {
    switch (type.toLowerCase()) {
      case 'function':
        return <Code2 className="h-4 w-4" />;
      case 'class':
        return <Component className="h-4 w-4" />;
      case 'variable':
        return <Box className="h-4 w-4" />;
      case 'module':
        return <Boxes className="h-4 w-4" />;
      case 'file':
        return <File className="h-4 w-4" />;
      case 'directory':
        return <FolderTree className="h-4 w-4" />;
      case 'database':
        return <Database className="h-4 w-4" />;
      default:
        return <FileCode className="h-4 w-4" />;
    }
  };
  
  const hasIssues = node.issues && node.issues.length > 0;
  
  // Calculate issue severity distribution
  const issueDistribution = hasIssues ? {
    CRITICAL: node.issues.filter(i => i.severity === IssueSeverity.CRITICAL).length,
    HIGH: node.issues.filter(i => i.severity === IssueSeverity.HIGH).length,
    MEDIUM: node.issues.filter(i => i.severity === IssueSeverity.MEDIUM).length,
    LOW: node.issues.filter(i => i.severity === IssueSeverity.LOW).length,
    INFO: node.issues.filter(i => i.severity === IssueSeverity.INFO).length
  } : null;
  
  const totalIssues = hasIssues ? node.issues.length : 0;
  
  return (
    <Card 
      className={cn(
        "transition-all duration-200",
        isSelected ? "border-primary shadow-md" : "",
        className
      )}
    >
      <Collapsible
        open={isOpen}
        onOpenChange={setIsOpen}
        className="w-full"
      >
        <div className="flex items-start p-4">
          <div className="mr-3 mt-0.5">
            {getNodeTypeIcon(node.type)}
          </div>
          
          <div className="flex-1">
            <CollapsibleTrigger className="flex items-center justify-between w-full text-left">
              <div>
                <h3 className="font-medium">{node.name}</h3>
                <p className="text-sm text-muted-foreground">{node.path}</p>
              </div>
              {isOpen ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
            </CollapsibleTrigger>
          </div>
        </div>
        
        {showIssueCount && hasIssues && (
          <div className="px-4 pb-2">
            <Badge variant="outline" className="bg-red-100">
              {totalIssues} issues
            </Badge>
          </div>
        )}
        
        <CollapsibleContent>
          <CardContent className="pt-0 pb-4">
            <div className="space-y-4">
              {showIssueCount && hasIssues && issueDistribution && (
                <div>
                  <p className="text-sm font-medium mb-1">Issue severity</p>
                  <div className="flex h-2 overflow-hidden rounded-full bg-muted">
                    {issueDistribution.CRITICAL > 0 && (
                      <div 
                        className="bg-red-600" 
                        style={{ width: `${(issueDistribution.CRITICAL / totalIssues) * 100}%` }} 
                      />
                    )}
                    {issueDistribution.HIGH > 0 && (
                      <div 
                        className="bg-orange-500" 
                        style={{ width: `${(issueDistribution.HIGH / totalIssues) * 100}%` }} 
                      />
                    )}
                    {issueDistribution.MEDIUM > 0 && (
                      <div 
                        className="bg-yellow-500" 
                        style={{ width: `${(issueDistribution.MEDIUM / totalIssues) * 100}%` }} 
                      />
                    )}
                    {issueDistribution.LOW > 0 && (
                      <div 
                        className="bg-blue-500" 
                        style={{ width: `${(issueDistribution.LOW / totalIssues) * 100}%` }} 
                      />
                    )}
                    {issueDistribution.INFO > 0 && (
                      <div 
                        className="bg-gray-500" 
                        style={{ width: `${(issueDistribution.INFO / totalIssues) * 100}%` }} 
                      />
                    )}
                  </div>
                  <div className="flex justify-between text-xs text-muted-foreground mt-1">
                    <span>{issueDistribution.CRITICAL} critical</span>
                    <span>{issueDistribution.HIGH} high</span>
                    <span>{issueDistribution.MEDIUM} medium</span>
                    <span>{issueDistribution.LOW} low</span>
                  </div>
                </div>
              )}
              
              {showBlastRadius && node.blast_radius !== undefined && (
                <div>
                  <p className="text-sm font-medium mb-1">Blast Radius:</p>
                  <div className="flex items-center gap-2">
                    <Target className="h-4 w-4 text-orange-500" />
                    <span className="text-lg font-bold">{node.blast_radius}</span>
                    <span className="text-sm text-muted-foreground">nodes</span>
                  </div>
                </div>
              )}
              
              {showMetadata && node.metadata && Object.keys(node.metadata).length > 0 && (
                <div>
                  <p className="text-sm font-medium mb-1">Metadata:</p>
                  <div className="grid grid-cols-2 gap-2">
                    {Object.entries(node.metadata).map(([key, value]) => (
                      <div key={key} className="text-sm">
                        <span className="text-muted-foreground">{key}:</span>{' '}
                        <span className="font-medium">{value}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
              
              <div className="flex flex-wrap gap-2 mt-4">
                {onViewIssues && hasIssues && (
                  <Button 
                    variant="outline" 
                    size="sm" 
                    onClick={() => onViewIssues(node)}
                    className="flex items-center gap-1"
                  >
                    <AlertTriangle className="h-3 w-3" />
                    Issues
                  </Button>
                )}
                
                {onViewBlastRadius && (
                  <Button 
                    variant="outline" 
                    size="sm" 
                    onClick={() => onViewBlastRadius(node)}
                    className="flex items-center gap-1"
                  >
                    <Target className="h-3 w-3" />
                    Blast Radius
                  </Button>
                )}
                
                {onViewInExplorer && (
                  <Button 
                    variant="outline" 
                    size="sm" 
                    onClick={() => onViewInExplorer(node)}
                    className="flex items-center gap-1"
                  >
                    <ExternalLink className="h-3 w-3" />
                    Explorer
                  </Button>
                )}
                
                <Button 
                  variant={isSelected ? "default" : "outline"} 
                  size="sm" 
                  onClick={handleSelect}
                  className="flex items-center gap-1"
                >
                  {isSelected ? 'Deselect' : 'Select'}
                </Button>
              </div>
            </div>
          </CardContent>
        </CollapsibleContent>
      </Collapsible>
    </Card>
  );
}

