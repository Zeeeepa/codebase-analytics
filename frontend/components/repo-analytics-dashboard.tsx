"use client"

import { useState } from "react"
import { BarChart3, Code2, FileCode2, GitBranch, Github, Settings, MessageSquare, FileText, Code, RefreshCcw, PaintBucket, Brain, Search, ArrowLeft, Activity, Layers, TreePine, Database, Target } from "lucide-react"
import { Bar, BarChart, ResponsiveContainer, XAxis, YAxis } from "recharts"

import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"

const mockRepoData = {
  name: "vercel/next.js",
  description: "The React Framework for the Web",
  linesOfCode: 154321,
  cyclomaticComplexity: 15.7,
  depthOfInheritance: 3.2,
  halsteadVolume: 987654,
  maintainabilityIndex: 85,
  commentDensity: 18.5,
  sloc: 132456,
  lloc: 98765,
  numberOfFiles: 1200,
  numberOfFunctions: 4500,
  numberOfClasses: 300,
}

const mockCommitData = [
  { month: "October", commits: 130 },
  { month: "September", commits: 150 },
  { month: "August", commits: 120 },
  { month: "July", commits: 110 },
  { month: "June", commits: 140 },
  { month: "May", commits: 160 },
  { month: "April", commits: 170 },
  { month: "March", commits: 180 },
  { month: "February", commits: 190 },
  { month: "January", commits: 200 },
  { month: "December", commits: 210 },
  { month: "November", commits: 220 },
];

interface RepoAnalyticsResponse {
  repo_url: string;
  line_metrics: {
    total: {
      loc: number;
      lloc: number;
      sloc: number;
      comments: number;
      comment_density: number;
    }
  };
  cyclomatic_complexity: { 
    average: number;
    rank?: string;
  };
  depth_of_inheritance: { 
    average: number;
  };
  halstead_metrics: { 
    total_volume: number;
    average_volume: number;
    operators?: number;
    operands?: number;
  };
  maintainability_index: { 
    average: number;
    rank?: string;
  };
  description: string;
  num_files: number;
  num_functions: number;
  num_classes: number;
  num_symbols?: number;
  monthly_commits: Record<string, number>;
  // Enhanced analysis data
  codebase_summary?: string;
  file_analysis?: {
    total_files: number;
    analyzed_files: number;
    file_types: Record<string, number>;
  };
  function_analysis?: {
    total_functions: number;
    average_complexity: number;
    complex_functions: Array<{
      name: string;
      complexity: number;
      file: string;
    }>;
  };
  class_analysis?: {
    total_classes: number;
    inheritance_depth: number;
    classes_with_inheritance: number;
  };
  import_analysis?: {
    total_imports: number;
    external_modules: number;
    internal_imports: number;
    import_graph?: Array<{
      from: string;
      to: string;
      type: string;
    }>;
  };
}

interface RepoData {
  name: string;
  description: string;
  linesOfCode: number;
  cyclomaticComplexity: number;
  depthOfInheritance: number;
  halsteadVolume: number;
  maintainabilityIndex: number;
  commentDensity: number;
  sloc: number;
  lloc: number;
  numberOfFiles: number;
  numberOfFunctions: number;
  numberOfClasses: number;
}

export default function RepoAnalyticsDashboard() {
  const [repoUrl, setRepoUrl] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [showAnalysis, setShowAnalysis] = useState(false)
  const [analysisData, setAnalysisData] = useState<RepoAnalyticsResponse | null>(null)

  const parseRepoUrl = (input: string): string => {
    if (input.includes('github.com')) {
      const url = new URL(input);
      const pathParts = url.pathname.split('/').filter(Boolean);
      if (pathParts.length >= 2) {
        return `${pathParts[0]}/${pathParts[1]}`;
      }
    }
    return input;
  };

  const handleAnalyze = async () => {
    if (!repoUrl.trim()) {
      setError("Please enter a repository URL or name");
      return;
    }

    setIsLoading(true);
    setError(null);
    
    try {
      const response = await fetch('http://localhost:9998/analyze_repo', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          repo_url: repoUrl.trim()
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      
      if (data.error) {
        throw new Error(data.error);
      }

      setAnalysisData(data);
      setShowAnalysis(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to analyze repository. Please check the URL and try again.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleBack = () => {
    setShowAnalysis(false);
    setAnalysisData(null);
    setError(null);
  };

  const handleKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      handleAnalyze(); 
    }
  };

  return (
    <div className="min-h-screen bg-background text-foreground">
      {!showAnalysis ? (
        <div className="flex flex-col items-center justify-center min-h-screen p-4">
          <div className="text-center mb-8">
            <h1 className="text-4xl font-bold flex items-center justify-center gap-3 mb-4">
              <Search className="h-12 w-12 text-primary" />
              <span>Codebase Analytics</span>
            </h1>
            <p className="text-muted-foreground text-lg">Analyze any GitHub repository with comprehensive metrics</p>
          </div>
          <div className="flex items-center gap-3 w-full max-w-lg">
            <Input
              type="text"
              placeholder="Enter GitHub repository URL or owner/repo"
              value={repoUrl}
              onChange={(e) => setRepoUrl(e.target.value)}
              onKeyPress={handleKeyPress}
              className="flex-1 text-lg py-3"
              title="Format: https://github.com/owner/repo or owner/repo"
            />
            <Button 
              onClick={handleAnalyze} 
              disabled={isLoading}
              size="lg"
              className="px-8"
            >
              {isLoading ? "Analyzing..." : "Analyze"}
            </Button>
          </div>
          {error && (
            <div className="mt-4 p-4 bg-destructive/10 border border-destructive/20 rounded-lg max-w-lg w-full">
              <p className="text-destructive text-sm">{error}</p>
            </div>
          )}
          <footer className="absolute bottom-0 w-full text-center text-xs text-muted-foreground py-4">
            built with <a href="https://codegen.com" target="_blank" rel="noopener noreferrer" className="hover:text-primary">Codegen</a>
          </footer>
        </div>
      ) : (
        <div className="flex flex-col min-h-screen">
          {/* Header */}
          <header className="sticky top-0 z-10 border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
            <div className="w-full px-6 py-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-4">
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={handleBack}
                    className="flex items-center gap-2"
                  >
                    <ArrowLeft className="h-4 w-4" />
                    Back
                  </Button>
                  <div>
                    <h1 className="text-xl font-bold">Repository Analysis</h1>
                    <p className="text-sm text-muted-foreground">{analysisData?.repo_url}</p>
                  </div>
                </div>
                {isLoading && (
                  <div className="flex items-center gap-2">
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-primary" />
                    <span className="text-sm">Analyzing...</span>
                  </div>
                )}
              </div>
            </div>
          </header>
          {/* Main Content */}
          <main className="p-6 flex-grow">
            {analysisData && (
              <div className="space-y-6">
                {/* Repository Overview */}
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Github className="h-5 w-5" />
                      Repository Overview
                    </CardTitle>
                    <CardDescription>{analysisData.description}</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      <div className="text-center">
                        <div className="text-2xl font-bold text-primary">{analysisData.num_files?.toLocaleString() || 0}</div>
                        <div className="text-sm text-muted-foreground">Files</div>
                      </div>
                      <div className="text-center">
                        <div className="text-2xl font-bold text-primary">{analysisData.num_functions?.toLocaleString() || 0}</div>
                        <div className="text-sm text-muted-foreground">Functions</div>
                      </div>
                      <div className="text-center">
                        <div className="text-2xl font-bold text-primary">{analysisData.num_classes?.toLocaleString() || 0}</div>
                        <div className="text-sm text-muted-foreground">Classes</div>
                      </div>
                      <div className="text-center">
                        <div className="text-2xl font-bold text-primary">{analysisData.num_symbols?.toLocaleString() || 0}</div>
                        <div className="text-sm text-muted-foreground">Symbols</div>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                {/* Analysis Grid */}
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                  
                  {/* Line Metrics Analysis */}
                  <Card>
                    <CardHeader>
                      <CardTitle className="flex items-center gap-2">
                        <FileText className="h-5 w-5" />
                        Line Metrics Analysis
                      </CardTitle>
                      <CardDescription>Code line counting and density metrics</CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <div className="space-y-2">
                        <div className="flex justify-between">
                          <span className="text-sm">Total Lines (LOC)</span>
                          <span className="font-mono text-sm">{analysisData.line_metrics?.total.loc?.toLocaleString() || 0}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-sm">Logical Lines (LLOC)</span>
                          <span className="font-mono text-sm">{analysisData.line_metrics?.total.lloc?.toLocaleString() || 0}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-sm">Source Lines (SLOC)</span>
                          <span className="font-mono text-sm">{analysisData.line_metrics?.total.sloc?.toLocaleString() || 0}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-sm">Comments</span>
                          <span className="font-mono text-sm">{analysisData.line_metrics?.total.comments?.toLocaleString() || 0}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-sm">Comment Density</span>
                          <span className="font-mono text-sm">{analysisData.line_metrics?.total.comment_density}%</span>
                        </div>
                      </div>
                    </CardContent>
                  </Card>

                  {/* Cyclomatic Complexity Analysis */}
                  <Card>
                    <CardHeader>
                      <CardTitle className="flex items-center gap-2">
                        <Brain className="h-5 w-5" />
                        Cyclomatic Complexity
                      </CardTitle>
                      <CardDescription>Code complexity and maintainability metrics</CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <div className="text-center">
                        <div className="text-3xl font-bold text-primary">{analysisData.cyclomatic_complexity.average}</div>
                        <div className="text-sm text-muted-foreground">Average Complexity</div>
                      </div>
                      <div className="text-center">
                        <div className={`inline-flex px-3 py-1 rounded-full text-sm font-medium ${
                          analysisData.cyclomatic_complexity.rank === 'Low' ? 'bg-green-100 text-green-800' :
                          analysisData.cyclomatic_complexity.rank === 'Medium' ? 'bg-yellow-100 text-yellow-800' :
                          'bg-red-100 text-red-800'
                        }`}>
                          {analysisData.cyclomatic_complexity.rank} Complexity
                        </div>
                      </div>
                      <div className="text-xs text-muted-foreground text-center">
                        Lower complexity indicates easier maintenance
                      </div>
                    </CardContent>
                  </Card>

                  {/* Maintainability Index */}
                  <Card>
                    <CardHeader>
                      <CardTitle className="flex items-center gap-2">
                        <Target className="h-5 w-5" />
                        Maintainability Index
                      </CardTitle>
                      <CardDescription>Overall code quality assessment</CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <div className="text-center">
                        <div className="text-3xl font-bold text-primary">{analysisData.maintainability_index?.average}</div>
                        <div className="text-sm text-muted-foreground">out of 100</div>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-2">
                        <div 
                          className="bg-primary h-2 rounded-full transition-all duration-500" 
                          style={{ width: `${analysisData.maintainability_index?.average}%` }}
                        ></div>
                      </div>
                      <div className="text-center">
                        <div className={`inline-flex px-3 py-1 rounded-full text-sm font-medium ${
                          analysisData.maintainability_index?.rank === 'Excellent' ? 'bg-green-100 text-green-800' :
                          analysisData.maintainability_index?.rank === 'Good' ? 'bg-blue-100 text-blue-800' :
                          'bg-orange-100 text-orange-800'
                        }`}>
                          {analysisData.maintainability_index?.rank}
                        </div>
                      </div>
                    </CardContent>
                  </Card>

                  {/* Halstead Metrics */}
                  <Card>
                    <CardHeader>
                      <CardTitle className="flex items-center gap-2">
                        <Activity className="h-5 w-5" />
                        Halstead Metrics
                      </CardTitle>
                      <CardDescription>Software complexity based on operators and operands</CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <div className="space-y-2">
                        <div className="flex justify-between">
                          <span className="text-sm">Total Volume</span>
                          <span className="font-mono text-sm">{analysisData.halstead_metrics?.total_volume?.toLocaleString() || 0}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-sm">Average Volume</span>
                          <span className="font-mono text-sm">{analysisData.halstead_metrics?.average_volume?.toLocaleString() || 0}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-sm">Operators</span>
                          <span className="font-mono text-sm">{analysisData.halstead_metrics?.operators?.toLocaleString() || 0}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-sm">Operands</span>
                          <span className="font-mono text-sm">{analysisData.halstead_metrics?.operands?.toLocaleString() || 0}</span>
                        </div>
                      </div>
                    </CardContent>
                  </Card>

                  {/* Depth of Inheritance */}
                  <Card>
                    <CardHeader>
                      <CardTitle className="flex items-center gap-2">
                        <Layers className="h-5 w-5" />
                        Inheritance Analysis
                      </CardTitle>
                      <CardDescription>Object-oriented design metrics</CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <div className="text-center">
                        <div className="text-3xl font-bold text-primary">{analysisData.depth_of_inheritance.average}</div>
                        <div className="text-sm text-muted-foreground">Average Depth</div>
                      </div>
                      <div className="space-y-2">
                        <div className="flex justify-between">
                          <span className="text-sm">Total Classes</span>
                          <span className="font-mono text-sm">{analysisData.class_analysis?.total_classes?.toLocaleString() || 0}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-sm">With Inheritance</span>
                          <span className="font-mono text-sm">{analysisData.class_analysis?.classes_with_inheritance?.toLocaleString() || 0}</span>
                        </div>
                      </div>
                    </CardContent>
                  </Card>

                  {/* File Structure Analysis */}
                  <Card>
                    <CardHeader>
                      <CardTitle className="flex items-center gap-2">
                        <Database className="h-5 w-5" />
                        File Structure Analysis
                      </CardTitle>
                      <CardDescription>Codebase organization and file types</CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <div className="space-y-2">
                        <div className="flex justify-between">
                          <span className="text-sm">Total Files</span>
                          <span className="font-mono text-sm">{analysisData.file_analysis?.total_files?.toLocaleString() || 0}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-sm">Analyzed Files</span>
                          <span className="font-mono text-sm">{analysisData.file_analysis?.analyzed_files?.toLocaleString() || 0}</span>
                        </div>
                      </div>
                      <div className="space-y-1">
                        <div className="text-sm font-medium">File Types:</div>
                        {Object.entries(analysisData.file_analysis?.file_types || {}).map(([ext, count]) => (
                          <div key={ext} className="flex justify-between text-sm">
                            <span className="font-mono">{ext}</span>
                            <span>{count}</span>
                          </div>
                        ))}
                      </div>
                    </CardContent>
                  </Card>

                </div>

                {/* Function Analysis Details */}
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Code2 className="h-5 w-5" />
                      Function Analysis Details
                    </CardTitle>
                    <CardDescription>Detailed function complexity analysis</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <div>
                        <h4 className="font-medium mb-3">Overview</h4>
                        <div className="space-y-2">
                          <div className="flex justify-between">
                            <span className="text-sm">Total Functions</span>
                            <span className="font-mono text-sm">{analysisData.function_analysis?.total_functions?.toLocaleString() || 0}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-sm">Average Complexity</span>
                            <span className="font-mono text-sm">{analysisData.function_analysis?.average_complexity}</span>
                          </div>
                        </div>
                      </div>
                      <div>
                        <h4 className="font-medium mb-3">Complex Functions</h4>
                        <div className="space-y-2">
                          {analysisData.function_analysis?.complex_functions.map((func, index) => (
                            <div key={index} className="flex justify-between text-sm">
                              <span className="font-mono truncate">{func.name}</span>
                              <span className="text-red-600 font-medium">{func.complexity}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                {/* Import Analysis */}
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <TreePine className="h-5 w-5" />
                      Import & Dependency Analysis
                    </CardTitle>
                    <CardDescription>Module relationships and dependencies</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <div>
                        <h4 className="font-medium mb-3">Import Statistics</h4>
                        <div className="space-y-2">
                          <div className="flex justify-between">
                            <span className="text-sm">Total Imports</span>
                            <span className="font-mono text-sm">{analysisData.import_analysis?.total_imports?.toLocaleString() || 0}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-sm">External Modules</span>
                            <span className="font-mono text-sm">{analysisData.import_analysis?.external_modules?.toLocaleString() || 0}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-sm">Internal Imports</span>
                            <span className="font-mono text-sm">{analysisData.import_analysis?.internal_imports?.toLocaleString() || 0}</span>
                          </div>
                        </div>
                      </div>
                      <div>
                        <h4 className="font-medium mb-3">Import Graph Sample</h4>
                        <div className="space-y-1">
                          {(analysisData.import_analysis?.import_graph || []).slice(0, 5).map((imp, index) => (
                            <div key={index} className="text-sm font-mono">
                              <span className="text-blue-600">{imp.from}</span>
                              <span className="mx-2">→</span>
                              <span className="text-green-600">{imp.to}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                {/* Monthly Commits Chart */}
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <BarChart3 className="h-5 w-5" />
                      Monthly Commit Activity
                    </CardTitle>
                    <CardDescription>Repository development activity over time</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <ResponsiveContainer width="100%" height={300}>
                      <BarChart data={Object.entries(analysisData.monthly_commits).map(([month, commits]) => ({ month, commits }))}>
                        <XAxis dataKey="month" />
                        <YAxis />
                        <Bar dataKey="commits" fill="hsl(var(--primary))" />
                      </BarChart>
                    </ResponsiveContainer>
                  </CardContent>
                </Card>

                {/* Summary */}
                <Card>
                  <CardHeader>
                    <CardTitle>Codebase Summary</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-muted-foreground">{analysisData.codebase_summary}</p>
                  </CardContent>
                </Card>

              </div>
            )}
          </main>
        </div>
      )}
    </div>
  )
}
