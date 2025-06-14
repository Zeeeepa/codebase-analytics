/**
 * Unified entry point for the Codebase Analytics frontend.
 * This component serves as the main application wrapper and provides a consistent interface.
 */

"use client"

import { useState } from 'react'
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import CodebaseDashboard from "@/components/codebase-dashboard"
import { ThemeProvider } from "@/components/theme-provider"
import { DashboardProvider } from "@/components/dashboard-context"

export default function Main() {
  const [activeTab, setActiveTab] = useState("dashboard")

  return (
    <ThemeProvider
      attribute="class"
      defaultTheme="system"
      enableSystem
      disableTransitionOnChange
    >
      <DashboardProvider>
        <div className="container mx-auto py-4">
          <header className="mb-6">
            <h1 className="text-3xl font-bold tracking-tight">Codebase Analytics</h1>
            <p className="text-muted-foreground">
              Comprehensive codebase analysis and visualization platform
            </p>
          </header>

          <Tabs defaultValue="dashboard" value={activeTab} onValueChange={setActiveTab}>
            <TabsList className="mb-4 w-full max-w-md">
              <TabsTrigger value="dashboard">Dashboard</TabsTrigger>
              <TabsTrigger value="explorer">Explorer</TabsTrigger>
              <TabsTrigger value="analytics">Analytics</TabsTrigger>
              <TabsTrigger value="settings">Settings</TabsTrigger>
            </TabsList>
            
            <TabsContent value="dashboard" className="space-y-4">
              <CodebaseDashboard />
            </TabsContent>
            
            <TabsContent value="explorer" className="space-y-4">
              <div className="rounded-lg border p-4">
                <h2 className="text-xl font-semibold mb-2">Codebase Explorer</h2>
                <p className="text-muted-foreground">
                  Explore your codebase structure, dependencies, and issues in detail.
                </p>
                {/* Explorer content would be loaded here */}
              </div>
            </TabsContent>
            
            <TabsContent value="analytics" className="space-y-4">
              <div className="rounded-lg border p-4">
                <h2 className="text-xl font-semibold mb-2">Analytics Dashboard</h2>
                <p className="text-muted-foreground">
                  View detailed analytics and metrics about your codebase.
                </p>
                {/* Analytics content would be loaded here */}
              </div>
            </TabsContent>
            
            <TabsContent value="settings" className="space-y-4">
              <div className="rounded-lg border p-4">
                <h2 className="text-xl font-semibold mb-2">Settings</h2>
                <p className="text-muted-foreground">
                  Configure your codebase analytics preferences.
                </p>
                {/* Settings content would be loaded here */}
              </div>
            </TabsContent>
          </Tabs>
        </div>
      </DashboardProvider>
    </ThemeProvider>
  )
}
