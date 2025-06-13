"use client"

import { useState } from "react"
import { DashboardProvider } from "./dashboard-context"
import { DashboardNavigation } from "./dashboard-navigation"
import { RepositoryInputForm } from "./repository-input-form"
import { MetricsView } from "./analytics-views/metrics-view"
import { ExplorerView } from "./analytics-views/explorer-view"
import { BlastRadiusView } from "./analytics-views/blast-radius-view"
import { StructureView } from "./analytics-views/structure-view"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { AlertTriangle } from "lucide-react"
import { useDashboard } from "./dashboard-context"
import { Separator } from "@/components/ui/separator"

function DashboardContent() {
  const { activeView, isLoading, error, repoUrl, repoData } = useDashboard();

  const renderView = () => {
    switch (activeView) {
      case 'metrics':
        return <MetricsView />;
      case 'explorer':
        return <ExplorerView />;
      case 'blast-radius':
        return <BlastRadiusView />;
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
        </aside>
        <main className="flex-1 p-6 overflow-auto">
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

export default function CodebaseDashboard() {
  return (
    <DashboardProvider>
      <DashboardContent />
    </DashboardProvider>
  );
}

