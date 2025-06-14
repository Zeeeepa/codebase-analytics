"use client"

import { useState, useEffect } from "react"
import { DashboardProviderEnhanced, useDashboardEnhanced } from "./dashboard-context-enhanced"
import { DashboardNavigation } from "./dashboard-navigation"
import { RepositoryInputForm } from "./repository-input-form"
import { MetricsView } from "./analytics-views/metrics-view"
import { ExplorerViewEnhanced } from "./analytics-views/explorer-view-enhanced"
import { BlastRadiusViewEnhanced } from "./analytics-views/blast-radius-view-enhanced"
import { StructureView } from "./analytics-views/structure-view"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { AlertTriangle, ArrowLeft, Settings, HelpCircle } from "lucide-react"
import { Button } from "@/components/ui/button"
import { 
  useSharedAnalysisState, 
  useIssueSelection, 
  useNodeSelection, 
  useFileSelection 
} from "@/hooks/useSharedAnalysisState"
import { Badge } from "@/components/ui/badge"
import { Separator } from "@/components/ui/separator"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"

function DashboardContent() {
  const { 
    activeView, 
    isLoading, 
    error, 
    repoUrl, 
    repoData,
    navigateBack,
    selectedSymbol,
    selectedFile,
    allIssues,
    allNodes
  } = useDashboardEnhanced();
  
  // Get shared state from hooks
  const { selectedIssue } = useIssueSelection();
  const { selectedNode } = useNodeSelection();
  const { selectedFilePath } = useFileSelection();
  
  // Local state for UI
  const [canGoBack, setCanGoBack] = useState(false);
  
  // Check if we can go back
  useEffect(() => {
    // This would be handled by the navigateBack function in a real implementation
    setCanGoBack(true);
  }, [activeView]);

  const renderView = () => {
    switch (activeView) {
      case 'metrics':
        return <MetricsView />;
      case 'explorer':
        return <ExplorerViewEnhanced />;
      case 'blast-radius':
        return <BlastRadiusViewEnhanced />;
      case 'structure':
        return <StructureView />;
      case 'issues':
        return <div>Issues View (Coming Soon)</div>;
      case 'dependencies':
        return <div>Dependencies View (Coming Soon)</div>;
      default:
        return <MetricsView />;
    }
  };
  
  // Get selection summary
  const getSelectionSummary = () => {
    const selections = [];
    
    if (selectedIssue) {
      selections.push(`Issue: ${selectedIssue.message.substring(0, 30)}${selectedIssue.message.length > 30 ? '...' : ''}`);
    }
    
    if (selectedNode) {
      selections.push(`Node: ${selectedNode.name}`);
    }
    
    if (selectedFilePath) {
      selections.push(`File: ${selectedFilePath.split('/').pop()}`);
    }
    
    if (selectedSymbol) {
      selections.push(`Symbol: ${selectedSymbol}`);
    }
    
    return selections;
  };
  
  const selectionSummary = getSelectionSummary();

  return (
    <div className="flex flex-col min-h-screen">
      <header className="sticky top-0 z-10 border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="w-full px-8 py-4">
          <div className="flex items-center justify-between">
            <div className="flex-shrink-0">
              <h1
                className="text-2xl font-bold flex items-center space-x-3 cursor-pointer"
                onClick={() => window.location.reload()}
              >
                <img src="cg.png" alt="CG Logo" className="h-8 w-8" />
                <span>Codebase Analytics</span>
              </h1>
            </div>
            <div className="flex items-center gap-3 ml-auto">
              <RepositoryInputForm />
            </div>
          </div>
        </div>
      </header>

      <div className="flex-1 flex">
        <aside className="w-64 border-r hidden md:block">
          <DashboardNavigation />
          
          {/* Selection Summary */}
          {selectionSummary.length > 0 && (
            <div className="p-4 border-t">
              <h3 className="text-sm font-medium mb-2">Current Selection</h3>
              <div className="space-y-2">
                {selectionSummary.map((selection, index) => (
                  <div key={index} className="text-xs bg-muted p-2 rounded-md">
                    {selection}
                  </div>
                ))}
              </div>
            </div>
          )}
          
          {/* Stats Summary */}
          <div className="p-4 border-t">
            <h3 className="text-sm font-medium mb-2">Analysis Summary</h3>
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div className="bg-muted p-2 rounded-md">
                <div className="text-muted-foreground">Issues</div>
                <div className="font-medium">{allIssues.length}</div>
              </div>
              <div className="bg-muted p-2 rounded-md">
                <div className="text-muted-foreground">Nodes</div>
                <div className="font-medium">{allNodes.length}</div>
              </div>
            </div>
          </div>
        </aside>
        
        <main className="flex-1 p-6 overflow-auto">
          {/* Navigation and Context Bar */}
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              {canGoBack && (
                <Button 
                  variant="ghost" 
                  size="sm"
                  onClick={() => navigateBack()}
                  className="flex items-center gap-1"
                >
                  <ArrowLeft className="h-4 w-4" />
                  Back
                </Button>
              )}
              
              <Badge variant="outline" className="text-xs">
                {activeView.charAt(0).toUpperCase() + activeView.slice(1)} View
              </Badge>
              
              {repoUrl && (
                <Badge variant="secondary" className="text-xs">
                  {repoUrl}
                </Badge>
              )}
            </div>
            
            <div className="flex items-center gap-2">
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button variant="ghost" size="icon">
                      <HelpCircle className="h-4 w-4" />
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>
                    <p>Help and Documentation</p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
              
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button variant="ghost" size="icon">
                      <Settings className="h-4 w-4" />
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>
                    <p>Settings</p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
            </div>
          </div>
          
          <Separator className="mb-4" />
          
          {error && (
            <Alert variant="destructive" className="mb-4">
              <AlertTriangle className="h-4 w-4" />
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}
          
          {renderView()}
        </main>
      </div>

      <footer className="w-full text-center text-xs text-muted-foreground py-4 border-t">
        built with <a href="https://codegen.com" target="_blank" rel="noopener noreferrer" className="hover:text-primary">Codegen</a>
      </footer>
    </div>
  );
}

export default function CodebaseDashboardEnhanced() {
  return (
    <DashboardProviderEnhanced>
      <DashboardContent />
    </DashboardProviderEnhanced>
  );
}

