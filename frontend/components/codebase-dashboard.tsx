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
    <div className="flex flex-col min-h-[calc(100vh-12rem)] rounded-lg border shadow-sm overflow-hidden">
      <header className="sticky top-0 z-10 border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="w-full px-4 py-3">
          <div className="flex items-center justify-between">
            <div className="flex-shrink-0">
              <h1
                className="text-xl font-bold flex items-center gap-3 cursor-pointer transition-colors hover:text-primary"
                onClick={() => window.location.reload()}
              >
                <div className="p-1.5 rounded-md bg-primary/10 text-primary">
                  <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="lucide lucide-code-2">
                    <path d="m18 16 4-4-4-4"/>
                    <path d="m6 8-4 4 4 4"/>
                    <path d="m14.5 4-5 16"/>
                  </svg>
                </div>
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
        <aside className="w-64 border-r hidden md:block bg-muted/20">
          <DashboardNavigation />
        </aside>
        <main className="flex-1 p-6 overflow-auto">
          {error && (
            <Alert variant="destructive" className="mb-6 animate-in fade-in-50 duration-300">
              <AlertTriangle className="h-4 w-4" />
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}
          
          {isLoading ? (
            <div className="space-y-4 animate-pulse">
              <div className="h-8 bg-muted rounded-md w-1/3"></div>
              <div className="h-24 bg-muted rounded-md"></div>
              <div className="h-32 bg-muted rounded-md"></div>
              <div className="h-24 bg-muted rounded-md"></div>
            </div>
          ) : (
            <div className="animate-in fade-in-50 duration-300">
              {renderView()}
            </div>
          )}
        </main>
      </div>

      <footer className="w-full text-center text-xs text-muted-foreground py-3 border-t bg-muted/10">
        built with <a href="https://codegen.com" target="_blank" rel="noopener noreferrer" className="text-primary hover:underline transition-all">Codegen</a>
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
