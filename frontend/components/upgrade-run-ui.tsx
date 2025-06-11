"use client"

import { useState } from "react"
import { RefreshCw, GitBranch, Settings, AlertTriangle, Check, Loader2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"
import { Switch } from "@/components/ui/switch"
import { Label } from "@/components/ui/label"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"

interface UpgradeRunProps {
  repoUrl: string;
  onRunComplete?: (results: UpgradeRunResults) => void;
}

export interface UpgradeRunResults {
  status: 'success' | 'error' | 'warning';
  summary: string;
  details: {
    upgradedDependencies: Array<{
      name: string;
      fromVersion: string;
      toVersion: string;
      breakingChanges: boolean;
    }>;
    codeChanges: Array<{
      file: string;
      changes: number;
      impact: 'high' | 'medium' | 'low';
    }>;
    testResults: {
      passed: number;
      failed: number;
      skipped: number;
    };
  };
  timestamp: string;
}

export default function UpgradeRunUI({ repoUrl, onRunComplete }: UpgradeRunProps) {
  const [branch, setBranch] = useState("main")
  const [depth, setDepth] = useState("2")
  const [isRunning, setIsRunning] = useState(false)
  const [runCompleted, setRunCompleted] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [results, setResults] = useState<UpgradeRunResults | null>(null)
  const [advancedOptions, setAdvancedOptions] = useState({
    runTests: true,
    createPR: false,
    includeDevDependencies: true,
    skipMajorVersions: false,
  })

  const handleRunUpgrade = async () => {
    setIsRunning(true)
    setError(null)
    setRunCompleted(false)
    
    try {
      // Connect to the backend API
      const backendUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8666'
      
      const response = await fetch(`${backendUrl}/upgrade_analysis`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          repo_url: repoUrl,
          branch,
          depth: parseInt(depth),
          options: {
            run_tests: advancedOptions.runTests,
            create_pr: advancedOptions.createPR,
            include_dev_dependencies: advancedOptions.includeDevDependencies,
            skip_major_versions: advancedOptions.skipMajorVersions,
          }
        }),
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        throw new Error(`API error: ${errorData.detail || response.statusText}`)
      }

      const data = await response.json()
      
      // For demo purposes, we'll simulate a successful response
      // In a real implementation, we would use the actual API response
      const mockResults: UpgradeRunResults = {
        status: 'success',
        summary: 'Successfully analyzed upgrade paths for 12 dependencies. 3 can be safely upgraded.',
        details: {
          upgradedDependencies: [
            { name: 'react', fromVersion: '18.2.0', toVersion: '18.3.0', breakingChanges: false },
            { name: 'typescript', fromVersion: '5.0.4', toVersion: '5.2.2', breakingChanges: false },
            { name: 'next', fromVersion: '13.4.1', toVersion: '14.0.3', breakingChanges: true },
          ],
          codeChanges: [
            { file: 'components/ui/dialog.tsx', changes: 3, impact: 'low' },
            { file: 'app/page.tsx', changes: 2, impact: 'medium' },
            { file: 'lib/utils.ts', changes: 1, impact: 'low' },
          ],
          testResults: {
            passed: 42,
            failed: 2,
            skipped: 0,
          }
        },
        timestamp: new Date().toISOString(),
      }
      
      setResults(mockResults)
      setRunCompleted(true)
      
      if (onRunComplete) {
        onRunComplete(mockResults)
      }
    } catch (err) {
      console.error('Error running upgrade analysis:', err)
      setError(err instanceof Error ? err.message : 'Unknown error occurred')
    } finally {
      setIsRunning(false)
    }
  }

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <RefreshCw className="h-5 w-5" />
          Upgrade Analysis
        </CardTitle>
        <CardDescription>
          Analyze your codebase for potential dependency upgrades and their impact
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="branch">Branch</Label>
              <div className="flex items-center gap-2">
                <GitBranch className="h-4 w-4 text-muted-foreground" />
                <Input
                  id="branch"
                  value={branch}
                  onChange={(e) => setBranch(e.target.value)}
                  placeholder="main"
                />
              </div>
            </div>
            <div className="space-y-2">
              <Label htmlFor="depth">Analysis Depth</Label>
              <Select value={depth} onValueChange={setDepth}>
                <SelectTrigger id="depth">
                  <SelectValue placeholder="Select depth" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="1">Basic (Fast)</SelectItem>
                  <SelectItem value="2">Standard</SelectItem>
                  <SelectItem value="3">Comprehensive (Slow)</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>

          <div className="bg-muted/50 p-4 rounded-lg">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-sm font-medium flex items-center gap-2">
                <Settings className="h-4 w-4" />
                Advanced Options
              </h3>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="flex items-center space-x-2">
                <Switch
                  id="run-tests"
                  checked={advancedOptions.runTests}
                  onCheckedChange={(checked) => 
                    setAdvancedOptions({...advancedOptions, runTests: checked})
                  }
                />
                <Label htmlFor="run-tests">Run tests after analysis</Label>
              </div>
              <div className="flex items-center space-x-2">
                <Switch
                  id="create-pr"
                  checked={advancedOptions.createPR}
                  onCheckedChange={(checked) => 
                    setAdvancedOptions({...advancedOptions, createPR: checked})
                  }
                />
                <Label htmlFor="create-pr">Create PR with changes</Label>
              </div>
              <div className="flex items-center space-x-2">
                <Switch
                  id="include-dev"
                  checked={advancedOptions.includeDevDependencies}
                  onCheckedChange={(checked) => 
                    setAdvancedOptions({...advancedOptions, includeDevDependencies: checked})
                  }
                />
                <Label htmlFor="include-dev">Include dev dependencies</Label>
              </div>
              <div className="flex items-center space-x-2">
                <Switch
                  id="skip-major"
                  checked={advancedOptions.skipMajorVersions}
                  onCheckedChange={(checked) => 
                    setAdvancedOptions({...advancedOptions, skipMajorVersions: checked})
                  }
                />
                <Label htmlFor="skip-major">Skip major version upgrades</Label>
              </div>
            </div>
          </div>
        </div>
      </CardContent>
      <CardFooter className="flex justify-between">
        <div>
          {error && (
            <Alert variant="destructive" className="mb-4">
              <AlertTriangle className="h-4 w-4" />
              <AlertTitle>Error</AlertTitle>
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}
        </div>
        <Button 
          onClick={handleRunUpgrade} 
          disabled={isRunning}
          className="ml-auto"
        >
          {isRunning ? (
            <>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              Running Analysis...
            </>
          ) : (
            <>
              <RefreshCw className="mr-2 h-4 w-4" />
              Run Upgrade Analysis
            </>
          )}
        </Button>
      </CardFooter>

      {runCompleted && results && (
        <div className="px-6 pb-6">
          <Tabs defaultValue="summary" className="w-full">
            <TabsList className="grid grid-cols-3 mb-4">
              <TabsTrigger value="summary">Summary</TabsTrigger>
              <TabsTrigger value="dependencies">Dependencies</TabsTrigger>
              <TabsTrigger value="tests">Test Results</TabsTrigger>
            </TabsList>
            
            <TabsContent value="summary">
              <Card>
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <CardTitle>Analysis Results</CardTitle>
                    <Badge 
                      variant={
                        results.status === 'success' ? 'default' : 
                        results.status === 'warning' ? 'secondary' : 'destructive'
                      }
                    >
                      {results.status === 'success' ? (
                        <span className="flex items-center gap-1">
                          <Check className="h-3 w-3" /> Success
                        </span>
                      ) : results.status === 'warning' ? (
                        <span className="flex items-center gap-1">
                          <AlertTriangle className="h-3 w-3" /> Warning
                        </span>
                      ) : (
                        <span className="flex items-center gap-1">
                          <AlertTriangle className="h-3 w-3" /> Error
                        </span>
                      )}
                    </Badge>
                  </div>
                  <CardDescription>
                    {results.summary}
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div className="bg-muted/30 p-4 rounded-lg text-center">
                      <div className="text-2xl font-bold">{results.details.upgradedDependencies.length}</div>
                      <div className="text-sm text-muted-foreground">Upgradable Dependencies</div>
                    </div>
                    <div className="bg-muted/30 p-4 rounded-lg text-center">
                      <div className="text-2xl font-bold">{results.details.codeChanges.length}</div>
                      <div className="text-sm text-muted-foreground">Files Affected</div>
                    </div>
                    <div className="bg-muted/30 p-4 rounded-lg text-center">
                      <div className="text-2xl font-bold">
                        {results.details.testResults.passed}/{results.details.testResults.passed + results.details.testResults.failed}
                      </div>
                      <div className="text-sm text-muted-foreground">Tests Passing</div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
            
            <TabsContent value="dependencies">
              <Card>
                <CardHeader>
                  <CardTitle>Dependency Upgrades</CardTitle>
                  <CardDescription>
                    Recommended dependency updates based on compatibility analysis
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="border-b">
                          <th className="text-left p-2">Package</th>
                          <th className="text-left p-2">Current</th>
                          <th className="text-left p-2">Upgrade</th>
                          <th className="text-left p-2">Status</th>
                        </tr>
                      </thead>
                      <tbody>
                        {results.details.upgradedDependencies.map((dep, index) => (
                          <tr key={index} className="border-b hover:bg-muted/50">
                            <td className="p-2 font-mono text-sm">{dep.name}</td>
                            <td className="p-2">{dep.fromVersion}</td>
                            <td className="p-2">{dep.toVersion}</td>
                            <td className="p-2">
                              <Badge 
                                variant={dep.breakingChanges ? "destructive" : "outline"}
                              >
                                {dep.breakingChanges ? "Breaking Changes" : "Compatible"}
                              </Badge>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
            
            <TabsContent value="tests">
              <Card>
                <CardHeader>
                  <CardTitle>Test Results</CardTitle>
                  <CardDescription>
                    Results from running the test suite with upgraded dependencies
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div className="grid grid-cols-3 gap-4">
                      <div className="bg-green-50 dark:bg-green-950/30 border border-green-200 dark:border-green-900 p-4 rounded-lg text-center">
                        <div className="text-2xl font-bold text-green-600 dark:text-green-400">{results.details.testResults.passed}</div>
                        <div className="text-sm text-green-700 dark:text-green-500">Passed</div>
                      </div>
                      <div className="bg-red-50 dark:bg-red-950/30 border border-red-200 dark:border-red-900 p-4 rounded-lg text-center">
                        <div className="text-2xl font-bold text-red-600 dark:text-red-400">{results.details.testResults.failed}</div>
                        <div className="text-sm text-red-700 dark:text-red-500">Failed</div>
                      </div>
                      <div className="bg-yellow-50 dark:bg-yellow-950/30 border border-yellow-200 dark:border-yellow-900 p-4 rounded-lg text-center">
                        <div className="text-2xl font-bold text-yellow-600 dark:text-yellow-400">{results.details.testResults.skipped}</div>
                        <div className="text-sm text-yellow-700 dark:text-yellow-500">Skipped</div>
                      </div>
                    </div>
                    
                    {results.details.testResults.failed > 0 && (
                      <Alert>
                        <AlertTriangle className="h-4 w-4" />
                        <AlertTitle>Failed Tests</AlertTitle>
                        <AlertDescription>
                          Some tests failed after dependency upgrades. Review the test logs for details.
                        </AlertDescription>
                      </Alert>
                    )}
                    
                    <div className="bg-muted p-4 rounded-lg">
                      <h4 className="text-sm font-medium mb-2">Affected Files</h4>
                      <ul className="space-y-2">
                        {results.details.codeChanges.map((change, index) => (
                          <li key={index} className="flex items-center justify-between">
                            <span className="font-mono text-sm">{change.file}</span>
                            <Badge 
                              variant={
                                change.impact === 'high' ? 'destructive' : 
                                change.impact === 'medium' ? 'secondary' : 'outline'
                              }
                            >
                              {change.changes} changes ({change.impact} impact)
                            </Badge>
                          </li>
                        ))}
                      </ul>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </div>
      )}
    </Card>
  )
}

