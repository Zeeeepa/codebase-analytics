"use client"

import { useState } from "react"
import { BarChart3, Code2, FileCode2, GitBranch, Github, Settings, MessageSquare, FileText, Code, RefreshCcw, PaintBucket, Brain } from "lucide-react"
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
  const [repoData, setRepoData] = useState(mockRepoData)
  const [hoveredCard, setHoveredCard] = useState<string | null>(null)
  const [commitData, setCommitData] = useState(mockCommitData)
  const [isLoading, setIsLoading] = useState(false)
  const [isLandingPage, setIsLandingPage] = useState(true)
  const [error, setError] = useState<string | null>(null)

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

  const handleFetchRepo = async () => {
    console.log("Fetching repo data...");
    
    const parsedRepoUrl = parseRepoUrl(repoUrl);
    console.log(parsedRepoUrl);
    
    setIsLoading(true);
    setIsLandingPage(false);
    setError(null);
    
    try {
      console.log("Fetching repo data...");
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:9998';
      const response = await fetch(`${apiUrl}/analyze_repo`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
        body: JSON.stringify({ repo_url: parsedRepoUrl }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data: RepoAnalyticsResponse = await response.json();
      
      setRepoData({
        name: parsedRepoUrl,
        description: data.description,
        linesOfCode: data.line_metrics.total.loc,
        cyclomaticComplexity: data.cyclomatic_complexity.average,
        depthOfInheritance: data.depth_of_inheritance.average,
        halsteadVolume: data.halstead_metrics.total_volume,
        maintainabilityIndex: data.maintainability_index.average,
        commentDensity: data.line_metrics.total.comment_density,
        sloc: data.line_metrics.total.sloc,
        lloc: data.line_metrics.total.lloc,
        numberOfFiles: data.num_files,
        numberOfFunctions: data.num_functions,
        numberOfClasses: data.num_classes,
      });

      const transformedCommitData = Object.entries(data.monthly_commits)
        .map(([date, commits]) => ({
          month: new Date(date).toLocaleString('default', { month: 'long' }),
          commits,
        }))
        .slice(0, 12)
        .reverse();

      setCommitData(transformedCommitData);
    } catch (error) {
      console.error('Error fetching repo data:', error);
      const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
      setError(`Failed to analyze repository: ${errorMessage}. Please check the URL and try again.`);
      setIsLandingPage(true);
    } finally {
      setIsLoading(false);
    }
  };

  const handleMouseEnter = (cardName: string) => {
    setHoveredCard(cardName)
  }

  const handleMouseLeave = () => {
    setHoveredCard(null)
  }

  const handleKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      handleFetchRepo(); 
    }
  }

function calculateCodebaseGrade(data: RepoData) {
  const { maintainabilityIndex } = data;
  
  if (maintainabilityIndex >= 90) return 'A+';
  if (maintainabilityIndex >= 85) return 'A';
  if (maintainabilityIndex >= 80) return 'A-';
  if (maintainabilityIndex >= 75) return 'B+';
  if (maintainabilityIndex >= 70) return 'B';
  if (maintainabilityIndex >= 65) return 'B-';
  if (maintainabilityIndex >= 60) return 'C+';
  if (maintainabilityIndex >= 55) return 'C';
  if (maintainabilityIndex >= 50) return 'C-';
  if (maintainabilityIndex >= 45) return 'D+';
  if (maintainabilityIndex >= 40) return 'D';
  return 'F';
}




  return (
    <div className="min-h-screen bg-background text-foreground">
      {isLandingPage ? (
        <div className="flex flex-col items-center justify-center min-h-screen p-4">
          <div className="text-center mb-8">
            <h1 className="text-4xl font-bold flex items-center justify-center gap-3 mb-4">
              <img src="cg.png" alt="CG Logo" className="h-12 w-12" />
              <span>Codebase Analytics</span>
            </h1>
            <p className="text-muted-foreground">Effortlessly calculate GitHub repository metrics in seconds</p>
          </div>
          <div className="flex items-center gap-3 w-full max-w-lg">
            <Input
              type="text"
              placeholder="Enter the GitHub repo link or owner/repo"
              value={repoUrl}
              onChange={(e) => setRepoUrl(e.target.value)}
              onKeyPress={handleKeyPress}
              className="flex-1"
              title="Format: https://github.com/owner/repo or owner/repo"
            />
            <Button 
              onClick={handleFetchRepo} 
              disabled={isLoading}
            >
              {isLoading ? "Loading..." : "Analyze"}
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
      ) : isLoading ? (
        <div className="flex flex-col items-center justify-center min-h-screen">
          <div className="text-center mb-8">
            <h2 className="text-2xl font-bold mb-4">Analyzing Repository</h2>
            <p className="text-muted-foreground mb-2">Analyzing: <span className="font-mono text-primary">{parseRepoUrl(repoUrl)}</span></p>
            <p className="text-muted-foreground">Please wait while we calculate codebase metrics...</p>
          </div>
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary" />
        </div>
      ) : (
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
                  <Input
                    type="text"
                    placeholder="Enter the GitHub repo link or owner/repo"
                    value={repoUrl}
                    onChange={(e) => setRepoUrl(e.target.value)}
                    onKeyPress={handleKeyPress}
                    className="w-[320px]"
                    title="Format: https://github.com/owner/repo or owner/repo"
                  />
                  <Button onClick={handleFetchRepo} disabled={isLoading}>
                    {isLoading ? "Loading..." : "Analyze"}
                  </Button>
                  <Button 
                    variant="outline" 
                    onClick={() => {
                      setIsLandingPage(true);
                      setRepoUrl("");
                      setError(null);
                    }}
                  >
                    New Analysis
                  </Button>
                </div>
              </div>
            </div>
          </header>
          <main className="p-6 flex-grow">
            <div className="grid mb-5 gap-6 grid-cols-1">
              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Repository</CardTitle>
                  <Github className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <a href={`https://github.com/${repoData.name}`} target="_blank" rel="noopener noreferrer" className="block">
                    <div className="text-2xl font-bold">{repoData.name}</div>
                    <p className="text-xs text-muted-foreground mt-1">{repoData.description}</p>
                  </a>
                  <div className="flex flex-wrap mt-4 gap-4">
                    <div className="flex items-center">
                      <FileCode2 className="h-4 w-4 text-muted-foreground mr-2" />
                      <span className="text-sm font-medium">{repoData.numberOfFiles.toLocaleString()} Files</span>
                    </div>
                    <div className="flex items-center">
                      <Code className="h-4 w-4 text-muted-foreground mr-2" />
                      <span className="text-sm font-medium">{repoData.numberOfFunctions.toLocaleString()} Functions</span>
                    </div>
                    <div className="flex items-center">
                      <BarChart3 className="h-4 w-4 text-muted-foreground mr-2" />
                      <span className="text-sm font-medium">{repoData.numberOfClasses.toLocaleString()} Classes</span>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
            <div className="grid gap-6 md:grid-cols-4 lg:grid-cols-4 xl:grid-cols-4">
              <Card onMouseEnter={() => handleMouseEnter('Maintainability Index')} onMouseLeave={handleMouseLeave}>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Maintainability Index</CardTitle>
                  <Settings className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">{repoData.maintainabilityIndex}</div>
                  <p className="text-xs text-muted-foreground mt-1">
                    {hoveredCard === 'Maintainability Index' ? 'This evaluates how easy it is to understand, modify, and maintain a codebase (ranging from 0 to 100).' : 'Code maintainability score (0-100)'}
                  </p>
                </CardContent>
              </Card>
              <Card onMouseEnter={() => handleMouseEnter('Cyclomatic Complexity')} onMouseLeave={handleMouseLeave}>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Cyclomatic Complexity</CardTitle>
                  <RefreshCcw className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">{repoData.cyclomaticComplexity.toFixed(1)}</div>
                  <p className="text-xs text-muted-foreground mt-1">
                    {hoveredCard === 'Cyclomatic Complexity' ? 'This measures the number of independent paths through a program\'s source code' : 'Average complexity score'}
                  </p>
                </CardContent>
              </Card>
              <Card onMouseEnter={() => handleMouseEnter('Halstead Volume')} onMouseLeave={handleMouseLeave}>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Halstead Volume</CardTitle>
                  <PaintBucket className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">{repoData.halsteadVolume.toLocaleString()}</div>
                  <p className="text-xs text-muted-foreground mt-1">
                    {hoveredCard === 'Halstead Volume' ? 'This quantifies the amount of information in a program by measuring the size and complexity of its code using operators and operands.' : 'Code volume metric'}
                  </p>
                </CardContent>
              </Card>
              <Card onMouseEnter={() => handleMouseEnter('Depth of Inheritance')} onMouseLeave={handleMouseLeave}>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Depth of Inheritance</CardTitle>
                  <GitBranch className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">{repoData.depthOfInheritance.toFixed(1)}</div>
                  <p className="text-xs text-muted-foreground mt-1">
                    {hoveredCard === 'Depth of Inheritance' ? 'This is the average measure of the number of classes that a class inherits from.' : 'Average inheritance depth'}
                  </p>
                </CardContent>
              </Card>
              <Card onMouseEnter={() => handleMouseEnter('Lines of Code')} onMouseLeave={handleMouseLeave}>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Lines of Code</CardTitle>
                  <Code2 className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">{repoData.linesOfCode.toLocaleString()}</div>
                  <p className="text-xs text-muted-foreground mt-1">
                    {hoveredCard === 'Lines of Code' ? 'This is the total number of lines of code within this codebase.' : 'Total lines in the repository'}
                  </p>
                </CardContent>
              </Card>
              <Card onMouseEnter={() => handleMouseEnter('SLOC')} onMouseLeave={handleMouseLeave}>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">SLOC</CardTitle>
                  <FileText className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">{repoData.sloc.toLocaleString()}</div>
                  <p className="text-xs text-muted-foreground mt-1">
                    {hoveredCard === 'SLOC' ? 'This is the number of textual lines of code within the codebase, ignoring whitespace and comments.' : 'Source Lines of Code'}
                  </p>
                </CardContent>
              </Card>
              <Card onMouseEnter={() => handleMouseEnter('LLOC')} onMouseLeave={handleMouseLeave}>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">LLOC</CardTitle>
                  <Brain className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">{repoData.lloc.toLocaleString()}</div>
                  <p className="text-xs text-muted-foreground mt-1">
                    {hoveredCard === 'LLOC' ? 'This is the number of lines of code that contribute to executable statements in the codebase.' : 'Logical Lines of Code'}
                  </p>
                </CardContent>
              </Card>
              <Card onMouseEnter={() => handleMouseEnter('Comment Density')} onMouseLeave={handleMouseLeave}>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Comment Density</CardTitle>
                  <MessageSquare className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">{repoData.commentDensity.toFixed(1)}%</div>
                  <p className="text-xs text-muted-foreground mt-1">
                    {hoveredCard === 'Comment Density' ? 'This is the percentage of the lines in the codebase that are comments.' : 'Percentage of comments in code'}
                  </p>
                </CardContent>
              </Card>
            </div>
            <Card className="mt-6">
              <CardHeader>
                <CardTitle>Monthly Commits</CardTitle>
                <CardDescription>Number of commits, batched by month over the past year</CardDescription>
              </CardHeader>
              <CardContent className="pt-4">
                <ResponsiveContainer width="100%" height={300}>
                <BarChart data={commitData}>
                    <XAxis dataKey="month" stroke="#888888" />
                    <YAxis stroke="#888888" />
                    <Bar dataKey="commits" fill="#2563eb" />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
            <div className="grid gap-6 md:grid-cols-2">
              <Card className="mt-6">
                <CardContent className="pt-5 flex justify-between items-center">
                  <div>
                    <CardTitle>Codebase Grade</CardTitle>
                    <CardDescription>Overall grade based on code metrics</CardDescription>
                  </div>
                  <div className="text-4xl font-bold text-right">
                    {calculateCodebaseGrade(repoData)}
                  </div>
                </CardContent>
              </Card>
              <Card className="mt-6">
                <CardContent className="pt-5 flex justify-between items-center">
                  <div>
                    <CardTitle>Codebase Complexity</CardTitle>
                    <CardDescription>Judgment based on size and complexity</CardDescription>
                  </div>
                  <div className="text-2xl font-bold text-right">
                  {repoData.numberOfFiles > 1000 ? "Large" : "Moderate"}
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Enhanced Analysis Sections */}
            <div className="mt-8 space-y-6">
              <h2 className="text-xl font-semibold mb-4">Detailed Code Analysis</h2>
              
              {/* Function Analysis */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Code className="h-5 w-5" />
                    Function Analysis
                  </CardTitle>
                  <CardDescription>
                    Comprehensive analysis of functions including complexity metrics
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div className="text-center">
                      <div className="text-2xl font-bold text-blue-600">{repoData.numberOfFunctions}</div>
                      <div className="text-sm text-muted-foreground">Total Functions</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-green-600">{repoData.cyclomaticComplexity.toFixed(1)}</div>
                      <div className="text-sm text-muted-foreground">Avg Complexity</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-purple-600">
                        {repoData.cyclomaticComplexity > 10 ? 'High' : repoData.cyclomaticComplexity > 5 ? 'Medium' : 'Low'}
                      </div>
                      <div className="text-sm text-muted-foreground">Complexity Rank</div>
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Class Analysis */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Settings className="h-5 w-5" />
                    Class & Symbol Analysis
                  </CardTitle>
                  <CardDescription>
                    Object-oriented design metrics and symbol analysis
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                    <div className="text-center">
                      <div className="text-2xl font-bold text-indigo-600">{repoData.numberOfClasses}</div>
                      <div className="text-sm text-muted-foreground">Total Classes</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-orange-600">{repoData.depthOfInheritance.toFixed(1)}</div>
                      <div className="text-sm text-muted-foreground">Inheritance Depth</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-teal-600">{(repoData.numberOfFunctions + repoData.numberOfClasses * 3).toLocaleString()}</div>
                      <div className="text-sm text-muted-foreground">Est. Symbols</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-pink-600">
                        {repoData.depthOfInheritance > 4 ? 'Deep' : repoData.depthOfInheritance > 2 ? 'Moderate' : 'Shallow'}
                      </div>
                      <div className="text-sm text-muted-foreground">Design Pattern</div>
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Halstead Metrics */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Brain className="h-5 w-5" />
                    Halstead Complexity Metrics
                  </CardTitle>
                  <CardDescription>
                    Software complexity based on operators and operands analysis
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div className="text-center">
                      <div className="text-2xl font-bold text-red-600">{repoData.halsteadVolume.toLocaleString()}</div>
                      <div className="text-sm text-muted-foreground">Total Volume</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-yellow-600">{Math.round(repoData.halsteadVolume / repoData.numberOfFunctions).toLocaleString()}</div>
                      <div className="text-sm text-muted-foreground">Avg Volume/Function</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-cyan-600">
                        {Math.round(repoData.linesOfCode / 50).toLocaleString()}
                      </div>
                      <div className="text-sm text-muted-foreground">Est. Operators</div>
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* File & Import Analysis */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <FileText className="h-5 w-5" />
                    File Structure & Import Analysis
                  </CardTitle>
                  <CardDescription>
                    Codebase organization and dependency analysis
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                    <div className="text-center">
                      <div className="text-2xl font-bold text-emerald-600">{repoData.numberOfFiles}</div>
                      <div className="text-sm text-muted-foreground">Total Files</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-violet-600">{Math.round(repoData.linesOfCode / repoData.numberOfFiles)}</div>
                      <div className="text-sm text-muted-foreground">Avg Lines/File</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-amber-600">{Math.round(repoData.numberOfFiles * 0.3)}</div>
                      <div className="text-sm text-muted-foreground">Est. Imports</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-rose-600">
                        {repoData.numberOfFiles > 500 ? 'Large' : repoData.numberOfFiles > 100 ? 'Medium' : 'Small'}
                      </div>
                      <div className="text-sm text-muted-foreground">Project Size</div>
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Maintainability Summary */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <RefreshCcw className="h-5 w-5" />
                    Maintainability Summary
                  </CardTitle>
                  <CardDescription>
                    Overall assessment of code maintainability and quality
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div className="flex justify-between items-center">
                      <span className="font-medium">Maintainability Index</span>
                      <span className="text-2xl font-bold text-green-600">{repoData.maintainabilityIndex}/100</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div 
                        className="bg-green-600 h-2 rounded-full" 
                        style={{ width: `${repoData.maintainabilityIndex}%` }}
                      ></div>
                    </div>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                      <div className="text-center">
                        <div className="font-semibold text-blue-600">Code Quality</div>
                        <div>{repoData.maintainabilityIndex > 80 ? 'Excellent' : repoData.maintainabilityIndex > 60 ? 'Good' : 'Needs Work'}</div>
                      </div>
                      <div className="text-center">
                        <div className="font-semibold text-purple-600">Documentation</div>
                        <div>{repoData.commentDensity > 15 ? 'Well Documented' : 'Sparse'}</div>
                      </div>
                      <div className="text-center">
                        <div className="font-semibold text-orange-600">Complexity</div>
                        <div>{repoData.cyclomaticComplexity < 10 ? 'Manageable' : 'Complex'}</div>
                      </div>
                      <div className="text-center">
                        <div className="font-semibold text-teal-600">Structure</div>
                        <div>{repoData.depthOfInheritance < 4 ? 'Clean' : 'Deep Hierarchy'}</div>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </main>
          <footer className="w-full text-center text-xs text-muted-foreground py-4">
          built with <a href="https://codegen.com" target="_blank" rel="noopener noreferrer" className="hover:text-primary">Codegen</a>
          </footer>
        </div>
      )}
    </div>
  )
}
