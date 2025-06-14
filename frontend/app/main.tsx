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
        <div className="min-h-screen bg-background">
          <div className="container mx-auto px-4 py-6">
            <header className="mb-8">
              <h1 className="text-4xl font-bold tracking-tight bg-gradient-to-r from-primary to-primary/70 bg-clip-text text-transparent">
                Codebase Analytics
              </h1>
              <p className="text-muted-foreground text-lg mt-2">
                Comprehensive codebase analysis and visualization platform
              </p>
            </header>

            <Tabs 
              defaultValue="dashboard" 
              value={activeTab} 
              onValueChange={setActiveTab}
              className="space-y-6"
            >
              <div className="bg-background/95 backdrop-blur-sm sticky top-0 z-10 py-2">
                <TabsList className="grid grid-cols-4 w-full max-w-2xl">
                  <TabsTrigger value="dashboard" className="text-sm md:text-base">Dashboard</TabsTrigger>
                  <TabsTrigger value="explorer" className="text-sm md:text-base">Explorer</TabsTrigger>
                  <TabsTrigger value="analytics" className="text-sm md:text-base">Analytics</TabsTrigger>
                  <TabsTrigger value="settings" className="text-sm md:text-base">Settings</TabsTrigger>
                </TabsList>
              </div>
              
              <TabsContent value="dashboard" className="space-y-4 animate-in fade-in-50 duration-300">
                <CodebaseDashboard />
              </TabsContent>
              
              <TabsContent value="explorer" className="space-y-4 animate-in fade-in-50 duration-300">
                <div className="rounded-lg border p-6 shadow-sm">
                  <h2 className="text-2xl font-semibold mb-3">Codebase Explorer</h2>
                  <p className="text-muted-foreground">
                    Explore your codebase structure, dependencies, and issues in detail.
                  </p>
                  {/* Explorer content would be loaded here */}
                </div>
              </TabsContent>
              
              <TabsContent value="analytics" className="space-y-4 animate-in fade-in-50 duration-300">
                <div className="rounded-lg border p-6 shadow-sm">
                  <h2 className="text-2xl font-semibold mb-3">Analytics Dashboard</h2>
                  <p className="text-muted-foreground">
                    View detailed analytics and metrics about your codebase.
                  </p>
                  {/* Analytics content would be loaded here */}
                </div>
              </TabsContent>
              
              <TabsContent value="settings" className="space-y-4 animate-in fade-in-50 duration-300">
                <div className="rounded-lg border p-6 shadow-sm">
                  <h2 className="text-2xl font-semibold mb-3">Settings</h2>
                  <p className="text-muted-foreground">
                    Configure your codebase analytics preferences.
                  </p>
                  {/* Settings content would be loaded here */}
                </div>
              </TabsContent>
            </Tabs>
          </div>
        </div>
      </DashboardProvider>
    </ThemeProvider>
  )
}
