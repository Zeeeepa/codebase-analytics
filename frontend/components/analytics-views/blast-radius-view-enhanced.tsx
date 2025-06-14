"use client"

import { useState, useEffect, useMemo } from 'react';
import { useDashboard } from '@/components/dashboard-context';
import { apiService } from '@/lib/api-service';
import { calculateBlastRadiusStats } from '@/lib/analysis-utils';
import { useSymbolSelection, useNodeSelection } from '@/hooks/useSharedAnalysisState';
import { Target, AlertTriangle, Zap, Network, FileCode, ArrowRight, ArrowLeft } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { NodeCard } from '@/components/ui/node-card';
import { VisualNode } from '@/lib/api-types';

export function BlastRadiusViewEnhanced() {
  const { 
    repoUrl, 
    blastRadiusData, 
    setBlastRadiusData, 
    isLoading, 
    setIsLoading, 
    error, 
    setError,
    selectedSymbol,
    setSelectedSymbol,
    setActiveView
  } = useDashboard();

  // Get shared state from hooks
  const { selectedSymbolName, setSelectedSymbolName } = useSymbolSelection();
  const { selectedNode, setSelectedNode } = useNodeSelection();
  
  // Local state
  const [activeTab, setActiveTab] = useState('overview');
  const [localSymbol, setLocalSymbol] = useState(selectedSymbol || selectedSymbolName || '');

  // Update local symbol when selectedSymbol changes
  useEffect(() => {
    if (selectedSymbol) {
      setLocalSymbol(selectedSymbol);
    } else if (selectedSymbolName) {
      setLocalSymbol(selectedSymbolName);
    }
  }, [selectedSymbol, selectedSymbolName]);

  // Calculate blast radius statistics
  const blastRadiusStats = useMemo(() => {
    return calculateBlastRadiusStats(blastRadiusData);
  }, [blastRadiusData]);

  // Handle blast radius analysis
  const handleBlastRadiusAnalysis = async () => {
    if (!localSymbol.trim()) {
      setError("Please enter a symbol name for blast radius analysis");
      return;
    }

    setIsLoading(true);
    setError("");
    
    try {
      const data = await apiService.analyzeBlastRadius(repoUrl, localSymbol);
      setBlastRadiusData(data);
      
      // Update shared state
      setSelectedSymbol(localSymbol);
      setSelectedSymbolName(localSymbol);
      
      // If a node is returned, select it
      if (data.target_symbol) {
        setSelectedNode(data.target_symbol);
      }
    } catch (error) {
      console.error('Error during blast radius analysis:', error);
      setError('Failed to analyze symbol impact. Please check the symbol name and try again.');
    } finally {
      setIsLoading(false);
    }
  };

  // Handle view in explorer
  const handleViewInExplorer = (node: VisualNode) => {
    // Navigate to explorer view
    setActiveView('explorer');
    
    // Ideally, we would set a filter to show only this node
    // This would require modifying the explorer view component
  };

  // Get impact level color
  const getImpactLevelColor = (score: number) => {
    if (score >= 8) return 'text-red-500';
    if (score >= 6) return 'text-orange-500';
    if (score >= 4) return 'text-yellow-500';
    if (score >= 2) return 'text-blue-500';
    return 'text-gray-500';
  };

  return (
    <div className="space-y-6">
      {/* Analysis Controls */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-xl flex items-center gap-2">
            <Target className="h-5 w-5 text-orange-500" />
            Blast Radius Analysis
          </CardTitle>
          <CardDescription>
            Analyze the impact of changing a specific symbol in your codebase
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col gap-4">
            <div className="flex gap-2">
              <Input
                placeholder="Enter symbol name (function, class, or variable)"
                value={localSymbol}
                onChange={(e) => setLocalSymbol(e.target.value)}
                className="flex-1"
                disabled={isLoading}
              />
              
              <Button 
                onClick={handleBlastRadiusAnalysis} 
                disabled={isLoading || !localSymbol.trim() || !repoUrl}
                className="flex items-center gap-2"
              >
                <Zap className="h-4 w-4" />
                {isLoading ? 'Analyzing...' : 'Analyze Impact'}
              </Button>
            </div>
            
            {error && (
              <Alert variant="destructive">
                <AlertTriangle className="h-4 w-4" />
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}
            
            {isLoading && (
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span>Analyzing blast radius...</span>
                  <span>Please wait</span>
                </div>
                <Progress value={undefined} className="w-full" />
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Results */}
      {blastRadiusData && !isLoading && (
        <div className="space-y-6">
          {/* Target Symbol */}
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-lg">Target Symbol</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex items-start gap-4">
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-2">
                    {blastRadiusData.target_symbol.type === 'function' && (
                      <Badge variant="outline" className="bg-purple-100">Function</Badge>
                    )}
                    {blastRadiusData.target_symbol.type === 'class' && (
                      <Badge variant="outline" className="bg-blue-100">Class</Badge>
                    )}
                    {blastRadiusData.target_symbol.type === 'variable' && (
                      <Badge variant="outline" className="bg-green-100">Variable</Badge>
                    )}
                    {blastRadiusData.target_symbol.type === 'module' && (
                      <Badge variant="outline" className="bg-yellow-100">Module</Badge>
                    )}
                    <h3 className="text-xl font-bold">{blastRadiusData.target_symbol.name}</h3>
                  </div>
                  <p className="text-sm text-muted-foreground mb-4">{blastRadiusData.target_symbol.path}</p>
                  
                  {blastRadiusData.target_symbol.issues && blastRadiusData.target_symbol.issues.length > 0 && (
                    <div className="mb-4">
                      <h4 className="text-sm font-medium mb-2">Issues:</h4>
                      <div className="space-y-1">
                        {blastRadiusData.target_symbol.issues.map((issue, index) => (
                          <div key={index} className="text-sm flex items-start gap-2">
                            <AlertTriangle className="h-4 w-4 text-orange-500 mt-0.5" />
                            <span>{issue.message}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
                
                <div className="w-64 border rounded-lg p-4 bg-muted/50">
                  <h4 className="text-sm font-medium mb-3">Impact Summary</h4>
                  <div className="space-y-3">
                    <div>
                      <div className="text-sm text-muted-foreground">Affected Nodes</div>
                      <div className="text-2xl font-bold">{blastRadiusStats?.affectedNodes || 0}</div>
                    </div>
                    <div>
                      <div className="text-sm text-muted-foreground">Impact Score</div>
                      <div className={`text-2xl font-bold ${getImpactLevelColor(blastRadiusStats?.impactScore || 0)}`}>
                        {blastRadiusStats?.impactScore || 0}/10
                      </div>
                    </div>
                    <div>
                      <div className="text-sm text-muted-foreground">Percentage Affected</div>
                      <div className="text-2xl font-bold">{blastRadiusStats?.percentageAffected?.toFixed(1) || 0}%</div>
                    </div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Tabs for different views */}
          <Tabs defaultValue="overview" value={activeTab} onValueChange={setActiveTab}>
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="overview" className="flex items-center gap-1">
                <Target className="h-4 w-4" />
                Overview
              </TabsTrigger>
              <TabsTrigger value="affected-nodes" className="flex items-center gap-1">
                <FileCode className="h-4 w-4" />
                Affected Nodes
              </TabsTrigger>
              <TabsTrigger value="impact-graph" className="flex items-center gap-1">
                <Network className="h-4 w-4" />
                Impact Graph
              </TabsTrigger>
            </TabsList>
            
            {/* Overview Tab */}
            <TabsContent value="overview" className="space-y-4">
              <Card>
                <CardHeader>
                  <CardTitle>Impact Analysis</CardTitle>
                  <CardDescription>
                    Summary of the potential impact of changing {blastRadiusData.target_symbol.name}
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-6">
                    {/* Impact Score Visualization */}
                    <div>
                      <h4 className="text-sm font-medium mb-2">Impact Score</h4>
                      <div className="relative h-8 bg-muted rounded-full overflow-hidden">
                        <div 
                          className="absolute top-0 left-0 h-full bg-gradient-to-r from-green-500 via-yellow-500 to-red-500"
                          style={{ width: '100%' }}
                        />
                        <div 
                          className="absolute top-0 left-0 h-full flex items-center justify-center"
                          style={{ 
                            left: `${(blastRadiusStats?.impactScore || 0) * 10}%`, 
                            transform: 'translateX(-50%)' 
                          }}
                        >
                          <div className="h-8 w-2 bg-black" />
                        </div>
                        <div className="absolute top-0 left-0 w-full h-full flex items-center justify-center">
                          <span className="text-sm font-bold text-white drop-shadow-md">
                            {blastRadiusStats?.impactScore || 0}/10
                          </span>
                        </div>
                      </div>
                      <div className="flex justify-between text-xs text-muted-foreground mt-1">
                        <span>Low Impact</span>
                        <span>Medium Impact</span>
                        <span>High Impact</span>
                      </div>
                    </div>
                    
                    {/* Affected Areas */}
                    <div>
                      <h4 className="text-sm font-medium mb-2">Affected Areas</h4>
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        <Card>
                          <CardContent className="pt-6">
                            <div className="text-2xl font-bold">{blastRadiusData.blast_radius.affected_files || 0}</div>
                            <p className="text-xs text-muted-foreground">Files</p>
                          </CardContent>
                        </Card>
                        <Card>
                          <CardContent className="pt-6">
                            <div className="text-2xl font-bold">{blastRadiusData.blast_radius.affected_functions || 0}</div>
                            <p className="text-xs text-muted-foreground">Functions</p>
                          </CardContent>
                        </Card>
                        <Card>
                          <CardContent className="pt-6">
                            <div className="text-2xl font-bold">{blastRadiusData.blast_radius.affected_classes || 0}</div>
                            <p className="text-xs text-muted-foreground">Classes</p>
                          </CardContent>
                        </Card>
                      </div>
                    </div>
                    
                    {/* Recommendations */}
                    <div>
                      <h4 className="text-sm font-medium mb-2">Recommendations</h4>
                      <div className="bg-muted p-4 rounded-lg">
                        <div className="flex items-start gap-2">
                          <div className="mt-0.5">
                            <AlertTriangle className="h-5 w-5 text-orange-500" />
                          </div>
                          <div>
                            <p className="font-medium">Testing Recommendation</p>
                            <p className="text-sm mt-1">
                              Changes to this symbol will affect {blastRadiusStats?.affectedNodes || 0} nodes. 
                              Ensure comprehensive tests are in place before modifying.
                            </p>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
            
            {/* Affected Nodes Tab */}
            <TabsContent value="affected-nodes" className="space-y-4">
              <ScrollArea className="h-[600px]">
                <div className="space-y-4">
                  {blastRadiusData.affected_nodes && blastRadiusData.affected_nodes.length > 0 ? (
                    blastRadiusData.affected_nodes.map((node) => (
                      <NodeCard
                        key={node.id}
                        node={node}
                        onViewInExplorer={handleViewInExplorer}
                      />
                    ))
                  ) : (
                    <div className="text-center py-8">
                      <p className="text-muted-foreground">No affected nodes found.</p>
                    </div>
                  )}
                </div>
              </ScrollArea>
            </TabsContent>
            
            {/* Impact Graph Tab */}
            <TabsContent value="impact-graph" className="space-y-4">
              <Card>
                <CardHeader>
                  <CardTitle>Impact Graph</CardTitle>
                  <CardDescription>
                    Visual representation of how changes propagate through the codebase
                  </CardDescription>
                </CardHeader>
                <CardContent className="h-[500px] flex items-center justify-center">
                  {blastRadiusData.impact_graph && 
                   blastRadiusData.impact_graph.nodes && 
                   blastRadiusData.impact_graph.nodes.length > 0 ? (
                    <div className="w-full h-full bg-muted rounded-lg flex items-center justify-center">
                      <p className="text-muted-foreground">
                        Graph visualization would be rendered here
                      </p>
                    </div>
                  ) : (
                    <div className="text-center py-8">
                      <p className="text-muted-foreground">No graph data available.</p>
                    </div>
                  )}
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </div>
      )}
    </div>
  );
}

